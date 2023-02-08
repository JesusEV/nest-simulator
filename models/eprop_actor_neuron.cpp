/*
 *  eprop_actor_neuron.cpp
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

#include "eprop_actor_neuron.h"

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

RecordablesMap< actor_neuron >actor_neuron::recordablesMap_;

template <>
void
RecordablesMap< actor_neuron >::create()
{
  insert_( names::target_rate, &actor_neuron::get_target_rate_ );
  // DEBUG II: the commented line below was used in the pattern generation task
  //insert_( names::learning_signal, &actor_neuron::get_learning_signal_ );
  insert_( names::learning_signal, &actor_neuron::get_last_ls_ );
  insert_( names::V_m, &actor_neuron::get_V_m_ );
  insert_( names::len_eprop_hist, &actor_neuron::get_eprop_history_len );
  insert_( names::len_ls_per_syn, &actor_neuron::get_ls_per_syn_len );
}
/* ----------------------------------------------------------------
 * Default constructors defining default parameters and state
 * ---------------------------------------------------------------- */

nest::actor_neuron::Parameters_::Parameters_()
  : tau_m_( 10.0 )                                  // ms
  , c_m_( 250.0 )                                   // pF
  , E_L_( 0.0 )                                   // mV
  , I_e_( 0.0 )                                     // pA
  , V_min_( -std::numeric_limits< double >::max() ) // relative E_L_-55.0-E_L_
  , t_start_ls_( 0.0 )                               // ms
  , regression_( true )
  , update_interval_reset_( true )
  , output_averaging_period_( 1.0 )
  , ce_( 1.0 )
  , ce_decay_( 1.0 )
{
}

nest::actor_neuron::State_::State_()
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
nest::actor_neuron::Parameters_::get(
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

  def< double >( d, names::output_averaging_period, output_averaging_period_ );
  def< double >( d, names::ce, ce_ );
  def< double >( d, names::ce_decay, ce_decay_ );
}

double
nest::actor_neuron::Parameters_::set(
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

  updateValue< double >( d, names::output_averaging_period, output_averaging_period_ );
  updateValue< double >( d, names::ce, ce_ );
  updateValue< double >( d, names::ce_decay, ce_decay_ );

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
nest::actor_neuron::State_::get(
  DictionaryDatum& d, const Parameters_& p ) const
{
  def< double >( d, names::target_rate, target_rate_); // target_rate
  def< double >( d, names::learning_signal, learning_signal_ );
  def< double >( d, names::V_m, y3_ + p.E_L_ ); // Membrane potential
}

void
nest::actor_neuron::State_::set(
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


nest::actor_neuron::Buffers_::Buffers_(
  actor_neuron& n )
  : logger_( n )
{
}

nest::actor_neuron::Buffers_::Buffers_(
  const Buffers_&,
  actor_neuron& n )
  : logger_( n )
{
}

/* ----------------------------------------------------------------
 * Default and copy constructor for node
 * ---------------------------------------------------------------- */

nest::actor_neuron::actor_neuron()
  : EpropArchivingNode()
  , P_()
  , S_()
  , B_( *this )
{
  recordablesMap_.create();
}

nest::actor_neuron::actor_neuron(
  const actor_neuron& n )
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
nest::actor_neuron::init_state_( const Node& proto )
{
  const actor_neuron& pr = downcast< actor_neuron >( proto );
  S_ = pr.S_;
}

void
nest::actor_neuron::init_buffers_()
{
  B_.delayed_rates_.clear(); // includes resize

  B_.normalization_rates_.clear(); // includes resize
  B_.logger_.reset(); // includes resize
  B_.spikes_.clear();   // includes resize
  B_.currents_.clear(); // includes resize
  ArchivingNode::clear_history();

  // Try a better location for this
  V_.rand_generator_.seed(0);
}

void
nest::actor_neuron::pre_run_hook()
{
  B_.logger_
    .init(); // ensures initialization in case mm connected after Simulate

  const double h = Time::get_resolution().get_ms();
  V_.P33_ = std::exp( -h / P_.tau_m_ );
  V_.P30_ = 1 / P_.c_m_ * ( 1 - V_.P33_ ) * P_.tau_m_;
  V_.step_start_ls_ = (get_update_interval_steps()-1);
  V_.step_start_output_averaging_ = V_.step_start_ls_ 
       - Time( Time::ms( std::max( P_.output_averaging_period_, 0.0 ) ) ).get_steps();
  V_.actor_signal_ = 0.;
}

/* ----------------------------------------------------------------
 * Update and event handling functions
 */

void
nest::actor_neuron::update_( Time const& origin,
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

  // allocate memory to store readout signal to be sent by rate events
  std::vector< double > actor_signal_buffer( buffer_size, 0.0 );

  std::vector< double > to_critic_signal_buffer( 2*buffer_size, 0.0 );

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

    // compute the readout signal
    double actor_signal = std::exp( S_.y3_ + P_.E_L_ );
    // write exp( readout signal ) into the buffer which is used to send it to the other error neurons
    // in case of a regression task we don't need this and therefore set it to zero
    actor_signal_buffer[ lag ] = actor_signal;

    // DEBUG: changed sign (see tf code)
    learning_signal_buffer[ n_entries*lag ] = origin.get_steps() + lag + 1;

    // compute normalized learning signal from values stored in state_buffer_
    // which now contains the correct normalization because in the meantime
    // the other readout neurons have sent their membrane potential
    // the entries of the state_buffer_ are
    
    double normalized_learning_signal=0.0;
    double pi_out = 0.0;
    double hot_encoded_action = 0.0;
    double ce_reg = 0.0;
    double temporal_diff_error = 0.0;
    double entropy_reg_factor = 0.0;

    // fill the state buffer with new values
    //if ( t_mod_T == V_.step_ls_ + 2 )
    if ( t_mod_T == 2 && origin.get_steps() > get_update_interval_steps())
    {
      long episode = kernel().simulation_manager.get_reward_based_eprop_episode();
      V_.action_taken_here_ = false;
      normalized_learning_signal = V_.normalized_ls_;
      temporal_diff_error = V_.td_error_;
      ce_reg = P_.ce_* P_.ce_decay_ / (episode + P_.ce_decay_);
      entropy_reg_factor = ce_reg * V_.action_prob_  * (std::log(V_.action_prob_) + V_.entropy_);

      V_.normalized_ls_ = 0.0;
      V_.td_error_ = 0.; 
      V_.entropy_ = 0.;
    }
    else
    {
      normalized_learning_signal = 0.0;
    }

    if ( t_mod_T > V_.step_start_output_averaging_ )//&& t_mod_T <= V_.step_ls_)
    {
      if (V_.mean_action_probs_.size() == 0)
      {
        V_.mean_action_probs_.resize(V_.current_action_probs_.size());
        std::fill(V_.mean_action_probs_.begin(), V_.mean_action_probs_.end(), 0.);
      } 

      double normalization_rate = B_.normalization_rates_.get_value( lag ) + V_.actor_signal_;
      for (double& p : V_.current_action_probs_)
        p /= normalization_rate; 

      for (unsigned i=0; i < V_.mean_action_probs_.size(); ++i )
        V_.mean_action_probs_[i] += V_.current_action_probs_[i];
    }
    else
    {
      B_.normalization_rates_.get_value( lag );
    }

    if ( t_mod_T == V_.step_start_ls_ )
    { 
      for (double& p : V_.mean_action_probs_)
        p /= P_.output_averaging_period_; 

      V_.entropy_ = 0.;
      for (const double& p : V_.mean_action_probs_)
          V_.entropy_ -= p * std::log(p);

      V_.normalized_ls_ = V_.mean_action_probs_[0]; 
      V_.action_prob_ = V_.mean_action_probs_[0]; 
      V_.action_taken_here_ = false;

      std::vector<int> indxs(V_.mean_action_probs_.size());
      std::iota(indxs.begin(),indxs.end(),0); //Initializing
      std::stable_sort( indxs.begin(), indxs.end(),
      [&](int i,int j) 
      {return V_.mean_action_probs_[i] < V_.mean_action_probs_[j];});

      std::sort (V_.mean_action_probs_.begin(), V_.mean_action_probs_.end()); 
      std::discrete_distribution<int> distribution( V_.mean_action_probs_.begin(),
                                                    V_.mean_action_probs_.end() );
      int hot_encoded_indx = distribution(V_.rand_generator_);      
      V_.action_taken_here_ = (indxs[hot_encoded_indx]==0);
      if (V_.action_taken_here_) V_.normalized_ls_ -= 1.;

      std::fill(V_.mean_action_probs_.begin(), V_.mean_action_probs_.end(), 0.);
    }

    if ( t_mod_T >= V_.step_start_output_averaging_ )
    {
      V_.actor_signal_ =  actor_signal;
    }
    else
    {
      V_.actor_signal_ = 0.0;
    }

    // write normalized learning signal into history. Use the previous time step:
    // origin.get_steps() + lag (without + 1) because of the buffering in
    // actor_signal_buffer
    // previously, the function write_readout_history was used for this purpose
    Time const& t_norm_ls = Time::step( origin.get_steps() + lag );
    const double t_norm_ls_ms = t_norm_ls.get_ms();
    rbeprop_history_.push_back( histentry_rbeprop( t_norm_ls_ms,
          0.0, normalized_learning_signal, temporal_diff_error, entropy_reg_factor, 0 ) );

    // store the normalized learning signal in the buffer which is send to
    // the recurrent neurons via the learning signal connection
    learning_signal_buffer[ n_entries*lag + 1 ] = normalized_learning_signal;
    learning_signal_buffer[ n_entries*lag + 2 ] = temporal_diff_error;
    learning_signal_buffer[ n_entries*lag + 3 ] = entropy_reg_factor;
    S_.y0_ = B_.currents_.get_value( lag ); // set new input current
    S_.target_rate_ =  B_.delayed_rates_.get_value( lag );

    B_.logger_.record_data( origin.get_steps() + lag );
  }

  V_.current_action_probs_.clear();
  V_.current_action_probs_.push_back( actor_signal_buffer[0] ); // Study this

  // send learning signal
  // TODO: it would be much more efficient to send this in larger batches
  RewardBasedLearningSignalConnectionEvent drve;
  drve.set_coeffarray( learning_signal_buffer );
  kernel().event_delivery_manager.send_secondary( *this, drve );
  // time one time step larger than t_mod_T_final because the readout has to
  // be sent one time step in advance so that the normalization can be computed
  // and the learning signal is ready as soon as the recall starts.
  
  // send readout signal only if this is a classification task
  // rate connection to connect to other readout neurons
  DelayedRateConnectionEvent actor_event;
  actor_event.set_coeffarray( actor_signal_buffer );
  kernel().event_delivery_manager.send_secondary( *this, actor_event );

  if (V_.action_taken_here_ == true )
  {
    SpikeEvent se;
    kernel().event_delivery_manager.send( *this, se, 0 );
    V_.action_taken_here_ = false;
  }

  return;
}

bool
nest::actor_neuron::is_eprop_readout()
    {
        return true;
    }

bool
nest::actor_neuron::is_eprop_critic()
{
  return false;
}

bool
nest::actor_neuron::is_eprop_actor()
{
  return true;
}

void
nest::actor_neuron::handle(
  DelayedRateConnectionEvent& e )
{
  assert( 0 <= e.get_rport() && e.get_rport() < 3 );
  const double weight = e.get_weight();
  const long delay = e.get_delay_steps();
  size_t i = 0;
  std::vector< unsigned int >::iterator it = e.begin();
  if ( e.get_rport() == READOUT_SIG - MIN_RATE_RECEPTOR )
  {
    // handle port for readout signal
    // The call to get_coeffvalue( it ) in this loop also advances the iterator it
    while ( it != e.end() )
    {
      double actor_signal = e.get_coeffvalue( it );
//      V_.state_buffer_ [ 2 ] += actor_signal;
      B_.normalization_rates_.add_value( delay + i, actor_signal ) ;
      V_.current_action_probs_.push_back( actor_signal );
      ++i;
    }
  }
}

void
nest::actor_neuron::handle(
  TemporalDiffErrorConnectionEvent& e )
{
  assert( 0 <= e.get_rport() && e.get_rport() < 1 );
  size_t i = 0;

  std::vector< unsigned int >::iterator it = e.begin();

  const Time stamp = e.get_stamp();
  double t_ms = stamp.get_ms();

  while ( it != e.end() )
  {
    double temp_diff_err = e.get_coeffvalue( it );
    V_.td_error_ = temp_diff_err; 
    ++i;
  }
}

void
nest::actor_neuron::handle( SpikeEvent& e )
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
nest::actor_neuron::handle( CurrentEvent& e )
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
nest::actor_neuron::handle( DataLoggingRequest& e )
{
  B_.logger_.handle( e );
}

} // namespace
