/*
 *  critic_neuron.cpp
 *
 *  This file is part of NEST.
 *
 *  Copyright (C) 2004 The NEST Initiative
 *
 *  NEST is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  NEST is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with NEST.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#include "eprop_critic_neuron.h"

// C++ includes:
#include <cmath> // in case we need isnan() // fabs
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>

// Includes from libnestutil:
#include "numerics.h"

// Includes from nestkernel:
#include "exceptions.h"
#include "kernel_manager.h"
#include "universal_data_logger_impl.h"

// Includes from sli:
#include "dict.h"
#include "dictutils.h"
#include "doubledatum.h"
#include "integerdatum.h"
#include "gauss_rate.h"

namespace nest
{

/* ----------------------------------------------------------------
 * Recordables map
 * ---------------------------------------------------------------- */

RecordablesMap< critic_neuron >critic_neuron::recordablesMap_;

template <>
void
RecordablesMap< critic_neuron >::create()
{
  insert_( names::target_rate, &critic_neuron::get_target_rate_ );
  // DEBUG II: the commented line below was used in the pattern generation task
  //insert_( names::learning_signal, &critic_neuron::get_learning_signal_ );
  insert_( names::learning_signal, &critic_neuron::get_last_ls_ );
  insert_( names::V_m, &critic_neuron::get_V_m_ );
  insert_( names::len_eprop_hist, &critic_neuron::get_eprop_history_len );
  insert_( names::len_ls_per_syn, &critic_neuron::get_ls_per_syn_len );
}
/* ----------------------------------------------------------------
 * Default constructors defining default parameters and state
 * ---------------------------------------------------------------- */

nest::critic_neuron::Parameters_::Parameters_()
  : tau_m_( 10.0 )                                  // ms
  , c_m_( 250.0 )                                   // pF
  , E_L_( 0.0 )                                   // mV
  , I_e_( 0.0 )                                     // pA
  , V_min_( -std::numeric_limits< double >::max() ) // relative E_L_-55.0-E_L_
  , t_start_ls_( 0.0 )                               // ms
  , regression_( true )
  , update_interval_reset_( true )
  , cv_( 0.0 )                               // ms
  , rb_gamma_( 0.95 )
{
}

nest::critic_neuron::State_::State_()
  : target_rate_( 0.0 )
  , learning_signal_( 0.0 )
  , y0_( 0.0 )
  , y3_( 0.0 )
{
}

/* ----------------------------------------------------------------
 * Parameter and state extractions and manipulation functions
 * ---------------------------------------------------------------- */

void
nest::critic_neuron::Parameters_::get(
  DictionaryDatum& d ) const
{
  def< double >( d, names::E_L, E_L_ ); // Resting potential
  def< double >( d, names::I_e, I_e_ );
  def< double >( d, names::V_min, V_min_ + E_L_ );
  def< double >( d, names::C_m, c_m_ );
  def< double >( d, names::tau_m, tau_m_ );
  def< double >( d, names::start, t_start_ls_ );
  def< bool >( d, names::regression, regression_ );
  def< bool >( d, names::update_interval_reset, update_interval_reset_ );

  def< double >( d, names::cv, cv_ );
  def< double >( d, names::rb_gamma, rb_gamma_ );
}

double
nest::critic_neuron::Parameters_::set(
  const DictionaryDatum& d )
{
  const double ELold = E_L_;
  updateValue< double >( d, names::E_L, E_L_ );
  const double delta_EL = E_L_ - ELold;
  if ( updateValue< double >( d, names::V_min, V_min_ ) )
  {
    V_min_ -= E_L_;
  }

  updateValue< double >( d, names::I_e, I_e_ );
  updateValue< double >( d, names::C_m, c_m_ );
  updateValue< double >( d, names::tau_m, tau_m_ );
  updateValue< double >( d, names::start, t_start_ls_ );
  updateValue< bool >( d, names::regression, regression_ );
  updateValue< bool >( d, names::update_interval_reset, update_interval_reset_ );

  updateValue< double >( d, names::cv, cv_ );
  updateValue< double >( d, names::rb_gamma, rb_gamma_ );

  if ( c_m_ <= 0 )
  {
    throw BadProperty( "Capacitance must be >0." );
  }

  if ( tau_m_ <= 0 )
  {
    throw BadProperty( "Membrane time constant must be > 0." );
  }
  return delta_EL;
}

void
nest::critic_neuron::State_::get(
  DictionaryDatum& d, const Parameters_& p ) const
{
  def< double >( d, names::target_rate, target_rate_); // target_rate
  def< double >( d, names::learning_signal, learning_signal_ );
  def< double >( d, names::V_m, y3_ + p.E_L_ ); // Membrane potential
}

void
nest::critic_neuron::State_::set(
  const DictionaryDatum& d, const Parameters_& p, double delta_EL)
{
  updateValue< double >( d, names::target_rate, target_rate_ ); // target_rate
  updateValue< double >( d, names::learning_signal, learning_signal_ );

  if ( updateValue< double >( d, names::V_m, y3_ ) )
  {
    y3_ -= p.E_L_;
  }
  else
  {
    y3_ -= delta_EL;
  }
}


nest::critic_neuron::Buffers_::Buffers_(
  critic_neuron& n )
  : logger_( n )
{
}

nest::critic_neuron::Buffers_::Buffers_(
  const Buffers_&,
  critic_neuron& n )
  : logger_( n )
{
}

/* ----------------------------------------------------------------
 * Default and copy constructor for node
 * ---------------------------------------------------------------- */

nest::critic_neuron::critic_neuron()
  : EpropArchivingNode()
  , P_()
  , S_()
  , B_( *this )
{
  recordablesMap_.create();
}

nest::critic_neuron::critic_neuron(
  const critic_neuron& n )
  : EpropArchivingNode( n )
  , P_( n.P_ )
  , S_( n.S_ )
  , B_( n.B_, *this )
{
}

/* ----------------------------------------------------------------
 * Node initialization functions
 * ---------------------------------------------------------------- */

void
nest::critic_neuron::init_state_( const Node& proto )
{
  const critic_neuron& pr = downcast< critic_neuron >( proto );
  S_ = pr.S_;
}

void
nest::critic_neuron::init_buffers_()
{
  B_.delayed_rates_.clear(); // includes resize

  B_.normalization_rates_.clear(); // includes resize
  B_.logger_.reset(); // includes resize
  B_.spikes_.clear();   // includes resize
  B_.currents_.clear(); // includes resize
  ArchivingNode::clear_history();
}

void
nest::critic_neuron::pre_run_hook()
{
  B_.logger_
    .init(); // ensures initialization in case mm connected after Simulate

  const double h = Time::get_resolution().get_ms();
  V_.P33_ = std::exp( -h / P_.tau_m_ );
  V_.P30_ = 1 / P_.c_m_ * ( 1 - V_.P33_ ) * P_.tau_m_;
  // Which of these variable id actually used?
  //V_.step_start_ls_ = Time( Time::ms( std::max( P_.t_start_ls_, 0.0 ) ) ).get_steps();
  V_.readout_signal_ = 0.;
  V_.target_signal_ = 0.;

  V_.step_start_ls_ = (get_update_interval_steps()-1);

  V_.prev1_V_c_ = 0.0;
  V_.prev2_V_c_ = 0.0;
  V_.prev3_V_c_ = 0.0;
  V_.prev4_V_c_ = 0.0;
  V_.V_c_ = 0.0;
}

/* ----------------------------------------------------------------
 * Update and event handling functions
 */

void
nest::critic_neuron::update_( Time const& origin,
  const long from,
  const long to)
{
  assert(
    to >= 0 && ( delay ) from < kernel().connection_manager.get_min_delay() );
  assert( from < to );

  const size_t buffer_size = kernel().connection_manager.get_min_delay();

  // allocate memory to store learning signal to be sent by learning signal events
  size_t n_entries = 4;
  std::vector< double > learning_signal_buffer( n_entries*buffer_size, 0.0 );
  std::vector< double > temporal_diff_error_buffer( buffer_size, 0.0 );

  for ( long lag = from; lag < to; ++lag )
  {
    // DEBUG: added reset after each T to be compatible with tf code
    int t_mod_T = ( origin.get_steps() + lag - 3 ) % get_update_interval_steps();
    long move = kernel().simulation_manager.get_reward_based_eprop_current_move();
    if (move == 0 && t_mod_T == 0)
    {
      S_.y3_ = 0.0;
      B_.spikes_.clear();   // includes resize
    }

    // DEBUG: introduced factor ( 1 - exp( -dt / tau_m ) ) for campatibility wit tf code
    S_.y3_ = V_.P30_ * ( S_.y0_ + P_.I_e_ ) + V_.P33_ * S_.y3_ + ( 1 - V_.P33_ ) * B_.spikes_.get_value( lag );
    S_.y3_ = ( S_.y3_ < P_.V_min_ ? P_.V_min_ : S_.y3_ );

    V_.prev4_V_c_ = V_.prev3_V_c_;
    V_.prev3_V_c_ = V_.prev2_V_c_;
    V_.prev2_V_c_ = V_.prev1_V_c_;
    V_.prev1_V_c_ = V_.V_c_;
    V_.V_c_ = S_.y3_ + P_.E_L_;

    double temporal_diff_error = V_.reward_ + P_.rb_gamma_ * V_.prev3_V_c_ - V_.prev4_V_c_;
    double normalized_learning_signal = 1.0;
    double Return = 0.0;
    double hot_encoded_id;
    double entropy = 0.0;

    learning_signal_buffer[ n_entries*lag ] = origin.get_steps() + lag + 1;


    if ( t_mod_T == 1 && origin.get_steps() > get_update_interval_steps())
    {
      temporal_diff_error_buffer[lag + 0] = temporal_diff_error;
    }

    // write normalized learning signal into history. Use the previous time step:
    // origin.get_steps() + lag (without + 1) because of the buffering in
    // readout_signal_buffer
    // previously, the function write_readout_history was used for this purpose
    Time const& t_norm_ls = Time::step( origin.get_steps() + lag );
    const double t_norm_ls_ms = t_norm_ls.get_ms();
    rbeprop_history_.push_back( histentry_rbeprop( t_norm_ls_ms,
          0.0, normalized_learning_signal, temporal_diff_error, 0.0, 0 ) );

    // store the normalized learning signal in the buffer which is send to
    // the recurrent neurons via the learning signal connection
    learning_signal_buffer[ n_entries*lag + 1 ] = normalized_learning_signal;
    S_.y0_ = B_.currents_.get_value( lag ); // set new input current
    S_.target_rate_ =  B_.delayed_rates_.get_value( lag );

    B_.logger_.record_data( origin.get_steps() + lag );
  }

  TemporalDiffErrorConnectionEvent tde_event;
  tde_event.set_coeffarray( temporal_diff_error_buffer );
  kernel().event_delivery_manager.send_secondary( *this, tde_event );

  // send learning signal
  // TODO: it would be much more efficient to send this in larger batches
  RewardBasedLearningSignalConnectionEvent drve;
  drve.set_coeffarray( learning_signal_buffer );
  kernel().event_delivery_manager.send_secondary( *this, drve );
  // time one time step larger than t_mod_T_final because the readout has to
  // be sent one time step in advance so that the normalization can be computed
  // and the learning signal is ready as soon as the recall starts.
  return;
}

bool
nest::critic_neuron::is_eprop_readout()
    {
        return true;
    }

bool
nest::critic_neuron::is_eprop_critic()
{
  return true;
}

bool
nest::critic_neuron::is_eprop_actor()
{
  return false;
}

void
nest::critic_neuron::handle(
  DelayedRateConnectionEvent& e )
{
  assert( 0 <= e.get_rport() && e.get_rport() < 3 );
  const double weight = e.get_weight();
  const long delay = e.get_delay_steps();
  size_t i = 0;
  std::vector< unsigned int >::iterator it = e.begin();

  while ( it != e.end() )
  {
    double reward = e.get_coeffvalue( it );
    V_.reward_ = reward;
    ++i;
  }
}

void
nest::critic_neuron::handle( SpikeEvent& e )
{
  assert( e.get_delay_steps() > 0 );

  // EX: We must compute the arrival time of the incoming spike
  //     explicity, since it depends on delay and offset within
  //     the update cycle.  The way it is done here works, but
  //     is clumsy and should be improved.
  B_.spikes_.add_value(
    e.get_rel_delivery_steps( kernel().simulation_manager.get_slice_origin() ),
    e.get_weight() * e.get_multiplicity() );
}

void
nest::critic_neuron::handle( CurrentEvent& e )
{
  assert( e.get_delay_steps() > 0 );

  const double c = e.get_current();
  const double w = e.get_weight();

  // add weighted current; HEP 2002-10-04
  B_.currents_.add_value(
    e.get_rel_delivery_steps( kernel().simulation_manager.get_slice_origin() ),
    w * c );
}

void
nest::critic_neuron::handle( DataLoggingRequest& e )
{
  B_.logger_.handle( e );
}

} // namespace
