/*
 *  eprop_iaf.cpp
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

// nest models
#include "eprop_iaf.h"

// C++
#include <limits>

// libnestutil
#include "dict_util.h"
#include "numerics.h"

// nestkernel
#include "exceptions.h"
#include "kernel_manager.h"
#include "nest_impl.h"
#include "universal_data_logger_impl.h"

// sli
#include "dictutils.h"

namespace nest
{

void
register_eprop_iaf( const std::string& name )
{
  register_node_model< eprop_iaf >( name );
}

/* ----------------------------------------------------------------
 * Recordables map
 * ---------------------------------------------------------------- */

RecordablesMap< eprop_iaf > eprop_iaf::recordablesMap_;

template <>
void
RecordablesMap< eprop_iaf >::create()
{
  insert_( names::learning_signal, &eprop_iaf::get_learning_signal_ );
  insert_( names::surrogate_gradient, &eprop_iaf::get_surrogate_gradient_ );
  insert_( names::V_m, &eprop_iaf::get_v_m_ );
}

/* ----------------------------------------------------------------
 * Default constructors for parameters, state, and buffers
 * ---------------------------------------------------------------- */

eprop_iaf::Parameters_::Parameters_()
  : C_m_( 250.0 )
  , c_reg_( 0.0 )
  , E_L_( -70.0 )
  , f_target_( 0.01 )
  , beta_( 1.0 )
  , gamma_( 0.3 )
  , I_e_( 0.0 )
  , regular_spike_arrival_( true )
  , surrogate_gradient_function_( "piecewise_linear" )
  , t_ref_( 2.0 )
  , tau_m_( 10.0 )
  , V_min_( -std::numeric_limits< double >::max() )
  , V_th_( -55.0 - E_L_ )
  , kappa_( 0.97 )
  , eprop_isi_trace_cutoff_( std::numeric_limits< long >::max() )
  , delay_rec_out_( 1 )
  , delay_out_rec_( 1 )
  , delay_total_( 1 )
{
}

eprop_iaf::State_::State_()
  : learning_signal_( 0.0 )
  , r_( 0 )
  , surrogate_gradient_( 0.0 )
  , i_in_( 0.0 )
  , v_m_( 0.0 )
  , z_( 0.0 )
  , z_in_( 0.0 )
{
}

eprop_iaf::Buffers_::Buffers_( eprop_iaf& n )
  : logger_( n )
{
}

eprop_iaf::Buffers_::Buffers_( const Buffers_&, eprop_iaf& n )
  : logger_( n )
{
}

/* ----------------------------------------------------------------
 * Getter and setter functions for parameters and state
 * ---------------------------------------------------------------- */

void
eprop_iaf::Parameters_::get( DictionaryDatum& d ) const
{
  def< double >( d, names::C_m, C_m_ );
  def< double >( d, names::c_reg, c_reg_ );
  def< double >( d, names::E_L, E_L_ );
  def< double >( d, names::f_target, f_target_ );
  def< double >( d, names::beta, beta_ );
  def< double >( d, names::gamma, gamma_ );
  def< double >( d, names::I_e, I_e_ );
  def< bool >( d, names::regular_spike_arrival, regular_spike_arrival_ );
  def< std::string >( d, names::surrogate_gradient_function, surrogate_gradient_function_ );
  def< double >( d, names::t_ref, t_ref_ );
  def< double >( d, names::tau_m, tau_m_ );
  def< double >( d, names::V_min, V_min_ + E_L_ );
  def< double >( d, names::V_th, V_th_ + E_L_ );
  def< double >( d, names::kappa, kappa_ );
  def< long >( d, names::eprop_isi_trace_cutoff, eprop_isi_trace_cutoff_ );

  double delay_rec_out_ms = Time( Time::step( delay_rec_out_ ) ).get_ms();
  def< double >( d, names::delay_rec_out, delay_rec_out_ms );
  double delay_out_rec_ms = Time( Time::step( delay_out_rec_ ) ).get_ms();
  def< double >( d, names::delay_out_rec, delay_out_rec_ms );
}

double
eprop_iaf::Parameters_::set( const DictionaryDatum& d, Node* node )
{
  // if leak potential is changed, adjust all variables defined relative to it
  const double ELold = E_L_;
  updateValueParam< double >( d, names::E_L, E_L_, node );
  const double delta_EL = E_L_ - ELold;

  V_th_ -= updateValueParam< double >( d, names::V_th, V_th_, node ) ? E_L_ : delta_EL;
  V_min_ -= updateValueParam< double >( d, names::V_min, V_min_, node ) ? E_L_ : delta_EL;

  updateValueParam< double >( d, names::C_m, C_m_, node );
  updateValueParam< double >( d, names::c_reg, c_reg_, node );

  if ( updateValueParam< double >( d, names::f_target, f_target_, node ) )
  {
    f_target_ /= 1000.0; // convert from spikes/s to spikes/ms
  }

  updateValueParam< double >( d, names::beta, beta_, node );
  updateValueParam< double >( d, names::gamma, gamma_, node );
  updateValueParam< double >( d, names::I_e, I_e_, node );
  updateValueParam< bool >( d, names::regular_spike_arrival, regular_spike_arrival_, node );
  updateValueParam< std::string >( d, names::surrogate_gradient_function, surrogate_gradient_function_, node );
  updateValueParam< double >( d, names::t_ref, t_ref_, node );
  updateValueParam< double >( d, names::tau_m, tau_m_, node );
  updateValueParam< double >( d, names::kappa, kappa_, node );
  updateValueParam< long >( d, names::eprop_isi_trace_cutoff, eprop_isi_trace_cutoff_, node );

  double delay_rec_out_ms = Time( Time::step( delay_rec_out_ ) ).get_ms();
  updateValueParam< double >( d, names::delay_rec_out, delay_rec_out_ms, node );
  delay_rec_out_ = Time( Time::ms( delay_rec_out_ms ) ).get_steps();

  double delay_out_rec_ms = Time( Time::step( delay_out_rec_ ) ).get_ms();
  updateValueParam< double >( d, names::delay_out_rec, delay_out_rec_ms, node );
  delay_out_rec_ = Time( Time::ms( delay_out_rec_ms ) ).get_steps();

  if ( C_m_ <= 0 )
  {
    throw BadProperty( "Membrane capacitance C_m > 0 required." );
  }

  if ( c_reg_ < 0 )
  {
    throw BadProperty( "Firing rate regularization prefactor c_reg ≥ 0 required." );
  }

  if ( f_target_ < 0 )
  {
    throw BadProperty( "Firing rate regularization target rate f_target ≥ 0 required." );
  }

  if ( tau_m_ <= 0 )
  {
    throw BadProperty( "Membrane time constant tau_m > 0 required." );
  }

  if ( t_ref_ < 0 )
  {
    throw BadProperty( "Refractory time t_ref ≥ 0 required." );
  }

  if ( V_th_ < V_min_ )
  {
    throw BadProperty( "Spike threshold voltage V_th ≥ minimal voltage V_min required." );
  }

  if ( kappa_ < 0.0 or kappa_ > 1.0 )
  {
    throw BadProperty( "Eligibility trace low-pass filter kappa from range [0, 1] required." );
  }

  if ( eprop_isi_trace_cutoff_ < 0 )
  {
    throw BadProperty( "Cutoff of integration of eprop trace between spikes eprop_isi_trace_cutoff ≥ 0 required." );
  }

  if ( delay_rec_out_ < 1 )
  {
    throw BadProperty( "Connection delay from recurrent to output neuron ≥ 1 required." );
  }

  if ( delay_out_rec_ < 1 )
  {
    throw BadProperty( "Broadcast delay of learning signals ≥ 1 required." );
  }

  delay_total_ = delay_rec_out_ + ( delay_out_rec_ - 1 );

  return delta_EL;
}

void
eprop_iaf::State_::get( DictionaryDatum& d, const Parameters_& p ) const
{
  def< double >( d, names::V_m, v_m_ + p.E_L_ );
  def< double >( d, names::surrogate_gradient, surrogate_gradient_ );
  def< double >( d, names::learning_signal, learning_signal_ );
}

void
eprop_iaf::State_::set( const DictionaryDatum& d, const Parameters_& p, double delta_EL, Node* node )
{
  v_m_ -= updateValueParam< double >( d, names::V_m, v_m_, node ) ? p.E_L_ : delta_EL;
}

/* ----------------------------------------------------------------
 * Default and copy constructor for node
 * ---------------------------------------------------------------- */

eprop_iaf::eprop_iaf()
  : EpropArchivingNodeRecurrent()
  , P_()
  , S_()
  , B_( *this )
{
  recordablesMap_.create();
}

eprop_iaf::eprop_iaf( const eprop_iaf& n )
  : EpropArchivingNodeRecurrent( n )
  , P_( n.P_ )
  , S_( n.S_ )
  , B_( n.B_, *this )
{
}

/* ----------------------------------------------------------------
 * Node initialization functions
 * ---------------------------------------------------------------- */

void
eprop_iaf::init_buffers_()
{
  B_.spikes_.clear();   // includes resize
  B_.currents_.clear(); // includes resize
  B_.logger_.reset();   // includes resize
}

void
eprop_iaf::pre_run_hook()
{
  B_.logger_.init(); // ensures initialization in case multimeter connected after Simulate

  V_.RefractoryCounts_ = Time( Time::ms( P_.t_ref_ ) ).get_steps();

  compute_surrogate_gradient = select_surrogate_gradient( P_.surrogate_gradient_function_ );

  // calculate the entries of the propagator matrix for the evolution of the state vector

  const double dt = Time::get_resolution().get_ms();

  V_.P_v_m_ = std::exp( -dt / P_.tau_m_ );
  V_.P_i_in_ = P_.tau_m_ / P_.C_m_ * ( 1.0 - V_.P_v_m_ );
  V_.P_z_in_ = P_.regular_spike_arrival_ ? 1.0 : 1.0 - V_.P_v_m_;

  if ( eprop_history_.empty() )
  {
    for ( long t = -P_.delay_total_; t < 0; ++t )
    {
      emplace_new_eprop_history_entry( t );
    }
  }
}

long
eprop_iaf::get_shift() const
{
  return offset_gen_ + delay_in_rec_;
}

bool
eprop_iaf::is_eprop_recurrent_node() const
{
  return true;
}

/* ----------------------------------------------------------------
 * Update function
 * ---------------------------------------------------------------- */

void
eprop_iaf::update( Time const& origin, const long from, const long to )
{
  for ( long lag = from; lag < to; ++lag )
  {
    const long t = origin.get_steps() + lag;

    S_.z_in_ = B_.spikes_.get_value( lag );

    S_.v_m_ = V_.P_i_in_ * S_.i_in_ + V_.P_z_in_ * S_.z_in_ + V_.P_v_m_ * S_.v_m_;
    S_.v_m_ -= P_.V_th_ * S_.z_;
    S_.v_m_ = std::max( S_.v_m_, P_.V_min_ );

    S_.z_ = 0.0;

    S_.surrogate_gradient_ =
      ( this->*compute_surrogate_gradient )( S_.r_, S_.v_m_, P_.V_th_, P_.V_th_, P_.beta_, P_.gamma_ );

    emplace_new_eprop_history_entry( t );

    write_surrogate_gradient_to_history( t, S_.surrogate_gradient_ );

    if ( S_.v_m_ >= P_.V_th_ and S_.r_ == 0 )
    {
      SpikeEvent se;
      kernel().event_delivery_manager.send( *this, se, lag );

      S_.z_ = 1.0;

      if ( V_.RefractoryCounts_ > 0 )
      {
        S_.r_ = V_.RefractoryCounts_;
      }
    }

    write_firing_rate_reg_to_history( t, S_.z_, P_.f_target_, P_.kappa_, P_.c_reg_ );

    S_.learning_signal_ = get_learning_signal_from_history( t, false );

    if ( S_.r_ > 0 )
    {
      --S_.r_;
    }

    S_.i_in_ = B_.currents_.get_value( lag ) + P_.I_e_;

    B_.logger_.record_data( t );
  }
}

/* ----------------------------------------------------------------
 * Event handling functions
 * ---------------------------------------------------------------- */

void
eprop_iaf::handle( SpikeEvent& e )
{
  assert( e.get_delay_steps() > 0 );

  B_.spikes_.add_value(
    e.get_rel_delivery_steps( kernel().simulation_manager.get_slice_origin() ), e.get_weight() * e.get_multiplicity() );
}

void
eprop_iaf::handle( CurrentEvent& e )
{
  assert( e.get_delay_steps() > 0 );

  B_.currents_.add_value(
    e.get_rel_delivery_steps( kernel().simulation_manager.get_slice_origin() ), e.get_weight() * e.get_current() );
}

void
eprop_iaf::handle( LearningSignalConnectionEvent& e )
{
  for ( auto it_event = e.begin(); it_event != e.end(); )
  {
    const long time_step = e.get_stamp().get_steps();
    const double weight = e.get_weight();
    const double error_signal = e.get_coeffvalue( it_event ); // get_coeffvalue advances iterator
    const double learning_signal = weight * error_signal;

    write_learning_signal_to_history( time_step, learning_signal, false );
  }
}

void
eprop_iaf::handle( DataLoggingRequest& e )
{
  B_.logger_.handle( e );
}

void
eprop_iaf::compute_gradient( const long t_spike,
  const long t_spike_previous,
  double& z_previous,
  double& z_bar,
  double& e_bar,
  double& epsilon,
  double& weight,
  const CommonSynapseProperties& cp,
  WeightOptimizer* optimizer )
{
  double e = 0.0;         // eligibility trace
  double z = 0.0;         // spiking variable
  double z_current = 1.0; // buffer containing the spike that triggered the current integration
  double psi = 0.0;       // surrogate gradient
  double L = 0.0;         // learning signal
  double grad = 0.0;      // gradient

  const EpropSynapseCommonProperties& ecp = static_cast< const EpropSynapseCommonProperties& >( cp );
  const auto optimize_each_step = ( *ecp.optimizer_cp_ ).optimize_each_step_;

  auto eprop_hist_it = get_eprop_history( t_spike_previous - 1 );

  const long t_compute_until = std::min( t_spike_previous + P_.eprop_isi_trace_cutoff_, t_spike );

  for ( long t = t_spike_previous; t < t_compute_until; ++t, ++eprop_hist_it )
  {
    z = z_previous;
    z_previous = z_current;
    z_current = 0.0;

    psi = eprop_hist_it->surrogate_gradient_;
    L = eprop_hist_it->learning_signal_;

    z_bar = V_.P_v_m_ * z_bar + V_.P_z_in_ * z;
    e = psi * z_bar;
    e_bar = P_.kappa_ * e_bar + ( 1.0 - P_.kappa_ ) * e;

    if ( optimize_each_step )
    {
      grad = L * e_bar;
      weight = optimizer->optimized_weight( *ecp.optimizer_cp_, t, grad, weight );
    }
    else
    {
      grad += L * e_bar;
    }
  }

  if ( not optimize_each_step )
  {
    weight = optimizer->optimized_weight( *ecp.optimizer_cp_, t_compute_until, grad, weight );
  }

  const int power = t_spike - ( t_spike_previous + P_.eprop_isi_trace_cutoff_ );

  if ( power > 0 )
  {
    z_bar *= std::pow( V_.P_v_m_, power );
    e_bar *= std::pow( P_.kappa_, power );
  }
}

void
eprop_iaf::compute_gradient( const long t_spike,
  const long t_spike_previous,
  std::queue< double >& z_previous_buffer,
  double& z_bar,
  double& e_bar,
  double& epsilon,
  double& weight,
  const CommonSynapseProperties& cp,
  WeightOptimizer* optimizer )
{
  double e = 0.0;    // eligibility trace
  double z = 0.0;    // spiking variable
  double psi = 0.0;  // surrogate gradient
  double L = 0.0;    // learning signal
  double grad = 0.0; // gradient

  const EpropSynapseCommonProperties& ecp = static_cast< const EpropSynapseCommonProperties& >( cp );
  const auto optimize_each_step = ( *ecp.optimizer_cp_ ).optimize_each_step_;  

  auto eprop_hist_it = get_eprop_history( t_spike_previous - P_.delay_total_ );

  const long t_compute_until = std::min( t_spike_previous + P_.eprop_isi_trace_cutoff_, t_spike );

  for ( long t = t_spike_previous; t < t_compute_until; ++t, ++eprop_hist_it )
  {
    if ( !z_previous_buffer.empty() )
    {
      z = z_previous_buffer.front();
      z_previous_buffer.pop();
    }

    if ( t_spike - t > 1 )
    {
      z_previous_buffer.push( 0.0 );
    }
    else
    {
      z_previous_buffer.push( 1.0 );
    }

    psi = eprop_hist_it->surrogate_gradient_;
    L = eprop_hist_it->learning_signal_;

    z_bar = V_.P_v_m_ * z_bar + V_.P_z_in_ * z;
    e = psi * z_bar;
    e_bar = P_.kappa_ * e_bar + ( 1.0 - P_.kappa_ ) * e;

    if ( optimize_each_step )
    {
      grad = L * e_bar;
      weight = optimizer->optimized_weight( *ecp.optimizer_cp_, t, grad, weight );
    }
    else
    {
      grad += L * e_bar;
    }
  }

  if ( not optimize_each_step )
  {
    weight = optimizer->optimized_weight( *ecp.optimizer_cp_, t_compute_until, grad, weight );
  }

  const int power = t_spike - ( t_spike_previous + P_.eprop_isi_trace_cutoff_ );

  if ( power > 0 )
  {
    z_bar *= std::pow( V_.P_v_m_, power );
    e_bar *= std::pow( P_.kappa_, power );
  }
}

} // namespace nest
