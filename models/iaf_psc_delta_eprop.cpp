/*
 *  iaf_psc_delta_eprop.cpp
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

/* iaf_psc_delta_eprop is a neuron where the potential jumps on each spike arrival. */

#include "iaf_psc_delta_eprop.h"

// C++ includes:
#include <limits>

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

namespace nest
{

/* ----------------------------------------------------------------
 * Recordables map
 * ---------------------------------------------------------------- */

RecordablesMap< iaf_psc_delta_eprop > iaf_psc_delta_eprop::recordablesMap_;

// Override the create() method with one call to RecordablesMap::insert_()
// for each quantity to be recorded.
template <>
void
RecordablesMap< iaf_psc_delta_eprop >::create()
{
  // use standard names whereever you can for consistency!
  insert_( names::V_m, &iaf_psc_delta_eprop::get_V_m_ );
  insert_( names::V_th, &iaf_psc_delta_eprop::get_last_h_ );
  insert_( names::E_L, &iaf_psc_delta_eprop::get_last_ls_ );
  insert_( names::len_eprop_hist, &iaf_psc_delta_eprop::get_eprop_history_len );
  insert_( names::len_ls_per_syn, &iaf_psc_delta_eprop::get_ls_per_syn_len );
}

/* ----------------------------------------------------------------
 * Default constructors defining default parameters and state
 * ---------------------------------------------------------------- */

nest::iaf_psc_delta_eprop::Parameters_::Parameters_()
  : tau_m_( 10.0 )                                  // ms
  , c_m_( 250.0 )                                   // pF
  , t_ref_( 2.0 )                                   // ms
  , E_L_( -70.0 )                                   // mV
  , I_e_( 0.0 )                                     // pA
  , V_th_( -55.0 - E_L_ )                           // mV, rel to E_L_
  , V_min_( -std::numeric_limits< double >::max() ) // relative E_L_-55.0-E_L_
  , V_reset_( -70.0 - E_L_ )                        // mV, rel to E_L_
  , with_refr_input_( false )
{
}

nest::iaf_psc_delta_eprop::State_::State_()
  : y0_( 0.0 )
  , y3_( 0.0 )
  , r_( 0 )
  , refr_spikes_buffer_( 0.0 )
{
}

/* ----------------------------------------------------------------
 * Parameter and state extractions and manipulation functions
 * ---------------------------------------------------------------- */

void
nest::iaf_psc_delta_eprop::Parameters_::get( DictionaryDatum& d ) const
{
  def< double >( d, names::E_L, E_L_ ); // Resting potential
  def< double >( d, names::I_e, I_e_ );
  def< double >( d, names::V_th, V_th_ + E_L_ ); // threshold value
  def< double >( d, names::V_reset, V_reset_ + E_L_ );
  def< double >( d, names::V_min, V_min_ + E_L_ );
  def< double >( d, names::C_m, c_m_ );
  def< double >( d, names::tau_m, tau_m_ );
  def< double >( d, names::t_ref, t_ref_ );
  def< bool >( d, names::refractory_input, with_refr_input_ );
}

double
nest::iaf_psc_delta_eprop::Parameters_::set( const DictionaryDatum& d )
{
  // if E_L_ is changed, we need to adjust all variables defined relative to
  // E_L_
  const double ELold = E_L_;
  updateValue< double >( d, names::E_L, E_L_ );
  const double delta_EL = E_L_ - ELold;

  if ( updateValue< double >( d, names::V_reset, V_reset_ ) )
  {
    V_reset_ -= E_L_;
  }
  else
  {
    V_reset_ -= delta_EL;
  }

  if ( updateValue< double >( d, names::V_th, V_th_ ) )
  {
    V_th_ -= E_L_;
  }
  else
  {
    V_th_ -= delta_EL;
  }

  if ( updateValue< double >( d, names::V_min, V_min_ ) )
  {
    V_min_ -= E_L_;
  }
  else
  {
    V_min_ -= delta_EL;
  }

  updateValue< double >( d, names::I_e, I_e_ );
  updateValue< double >( d, names::C_m, c_m_ );
  updateValue< double >( d, names::tau_m, tau_m_ );
  updateValue< double >( d, names::t_ref, t_ref_ );
  if ( V_reset_ >= V_th_ )
  {
    throw BadProperty( "Reset potential must be smaller than threshold." );
  }
  if ( c_m_ <= 0 )
  {
    throw BadProperty( "Capacitance must be >0." );
  }
  if ( t_ref_ < 0 )
  {
    throw BadProperty( "Refractory time must not be negative." );
  }
  if ( tau_m_ <= 0 )
  {
    throw BadProperty( "Membrane time constant must be > 0." );
  }

  updateValue< bool >( d, names::refractory_input, with_refr_input_ );

  return delta_EL;
}

void
nest::iaf_psc_delta_eprop::State_::get( DictionaryDatum& d,
  const Parameters_& p ) const
{
  def< double >( d, names::V_m, y3_ + p.E_L_ ); // Membrane potential
}

void
nest::iaf_psc_delta_eprop::State_::set( const DictionaryDatum& d,
  const Parameters_& p,
  double delta_EL )
{
  if ( updateValue< double >( d, names::V_m, y3_ ) )
  {
    y3_ -= p.E_L_;
  }
  else
  {
    y3_ -= delta_EL;
  }
}

nest::iaf_psc_delta_eprop::Buffers_::Buffers_( iaf_psc_delta_eprop& n )
  : logger_( n )
{
}

nest::iaf_psc_delta_eprop::Buffers_::Buffers_( const Buffers_&, iaf_psc_delta_eprop& n )
  : logger_( n )
{
}

/* ----------------------------------------------------------------
 * Default and copy constructor for node
 * ---------------------------------------------------------------- */

nest::iaf_psc_delta_eprop::iaf_psc_delta_eprop()
  : Eprop_Archiving_Node()
  , P_()
  , S_()
  , B_( *this )
{
  recordablesMap_.create();
}

nest::iaf_psc_delta_eprop::iaf_psc_delta_eprop( const iaf_psc_delta_eprop& n )
  : Eprop_Archiving_Node( n )
  , P_( n.P_ )
  , S_( n.S_ )
  , B_( n.B_, *this )
{
}

/* ----------------------------------------------------------------
 * Node initialization functions
 * ---------------------------------------------------------------- */

void
nest::iaf_psc_delta_eprop::init_state_( const Node& proto )
{
  const iaf_psc_delta_eprop& pr = downcast< iaf_psc_delta_eprop >( proto );
  S_ = pr.S_;
}

void
nest::iaf_psc_delta_eprop::init_buffers_()
{
  B_.spikes_.clear();   // includes resize
  B_.currents_.clear(); // includes resize
  B_.logger_.reset();   // includes resize
  init_eprop_buffers();
  Eprop_Archiving_Node::clear_history();
}

void
nest::iaf_psc_delta_eprop::calibrate()
{
  B_.logger_.init();

  const double h = Time::get_resolution().get_ms();


  V_.P33_ = std::exp( -h / P_.tau_m_ );
  V_.P30_ = 1 / P_.c_m_ * ( 1 - V_.P33_ ) * P_.tau_m_;


  // t_ref_ specifies the length of the absolute refractory period as
  // a double in ms. The grid based iaf_psp_delta can only handle refractory
  // periods that are integer multiples of the computation step size (h).
  // To ensure consistency with the overall simulation scheme such conversion
  // should be carried out via objects of class nest::Time. The conversion
  // requires 2 steps:
  //     1. A time object r is constructed, defining representation of
  //        t_ref_ in tics. This representation is then converted to computation
  //        time steps again by a strategy defined by class nest::Time.
  //     2. The refractory time in units of steps is read out get_steps(), a
  //        member function of class nest::Time.
  //
  // Choosing a t_ref_ that is not an integer multiple of the computation time
  // step h will lead to accurate (up to the resolution h) and self-consistent
  // results. However, a neuron model capable of operating with real valued
  // spike time may exhibit a different effective refractory time.

  V_.RefractoryCounts_ = Time( Time::ms( P_.t_ref_ ) ).get_steps();
  // since t_ref_ >= 0, this can only fail in error
  assert( V_.RefractoryCounts_ >= 0 );
}

/* ----------------------------------------------------------------
 * Update and spike handling functions
 */

void
nest::iaf_psc_delta_eprop::update( Time const& origin,
  const long from,
  const long to )
{
  assert(
    to >= 0 && ( delay ) from < kernel().connection_manager.get_min_delay() );
  assert( from < to );

  const double h = Time::get_resolution().get_ms();
  for ( long lag = from; lag < to; ++lag )
  {
    if ( S_.r_ == 0 )
    {
      // neuron not refractory
      // DEBUG: introduce factor ( 1 - exp( -dt / tau_m ) ) for incoming spikes
      S_.y3_ = V_.P30_ * ( S_.y0_ + P_.I_e_ ) + V_.P33_ * S_.y3_
        + ( 1.0 - V_.P33_ ) * B_.spikes_.get_value( lag );

      // if we have accumulated spikes from refractory period,
      // add and reset accumulator
      if ( P_.with_refr_input_ && S_.refr_spikes_buffer_ != 0.0 )
      {
        S_.y3_ += S_.refr_spikes_buffer_;
        S_.refr_spikes_buffer_ = 0.0;
      }

      // lower bound of membrane potential
      S_.y3_ = ( S_.y3_ < P_.V_min_ ? P_.V_min_ : S_.y3_ );
    }
    else // neuron is absolute refractory
    {
      // read spikes from buffer and accumulate them, discounting
      // for decay until end of refractory period
      if ( P_.with_refr_input_ )
      {
        S_.refr_spikes_buffer_ +=
          B_.spikes_.get_value( lag ) * std::exp( -S_.r_ * h / P_.tau_m_ );
      }
      else
      {
        B_.spikes_.get_value( lag );
      } // clear buffer entry, ignore spike

      --S_.r_;
    }

    // DEBUG: original implementation: write history after threshold crossing
    write_eprop_history( Time::step( origin.get_steps() + lag + 1 ), S_.y3_, P_.V_th_ );
    // threshold crossing
    if ( S_.y3_ >= P_.V_th_ )
    {
      S_.r_ = V_.RefractoryCounts_;
      // DEBUG: subtract threshold instead of setting to V_reset
      //S_.y3_ = P_.V_reset_;
      std::cout << S_.y3_ + P_.E_L_ << std::endl;
      std::cout << P_.V_th_ << ",  " << std::fabs( ( S_.y3_ - P_.V_th_ ) / P_.V_th_ ) << std::endl;
      S_.y3_ -= P_.V_th_;

      // EX: must compute spike time
      set_spiketime( Time::step( origin.get_steps() + lag + 1 ) );

      write_spike_history( Time::step( origin.get_steps() + lag + 1 ) );
      SpikeEvent se;
      kernel().event_delivery_manager.send( *this, se, lag );
    }

    // save learning signal for eprop algorithm
    // TODO: check if that are the quantities needed. My guess is that ist correct since 
    // in the paper the membrane potential is also measured wrt the resting potential.
    //write_eprop_history( Time::step( origin.get_steps() + lag + 1 ), get_V_m_(), P_.E_L_ + P_.V_th_ );

    // set new input current
    S_.y0_ = B_.currents_.get_value( lag );

    // voltage logging
    B_.logger_.record_data( origin.get_steps() + lag );
  }
}

double
nest::iaf_psc_delta_eprop::get_leak_propagator() const
{
  return V_.P33_;
}

bool
nest::iaf_psc_delta_eprop::is_eprop_readout()
{
  return false;
}

bool
nest::iaf_psc_delta_eprop::is_eprop_adaptive()
{
  return false;
}

void
nest::iaf_psc_delta_eprop::handle( SpikeEvent& e )
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
nest::iaf_psc_delta_eprop::handle( CurrentEvent& e )
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
nest::iaf_psc_delta_eprop::handle(
  DelayedRateConnectionEvent& e )
{
  /*
  const double weight = e.get_weight();
  const long delay = e.get_delay_steps();
  const Time stamp = e.get_stamp();

  std::vector< unsigned int >::iterator it = e.begin();
  */

  // Add learning signal to hist entries
  add_learning_to_hist( e );
  /*
  std::cout << "weight: " << weight << ", delay: " << delay << ", rate events: " << std::endl;
  while ( it != e.end() )
  {
    ++i;
    std::cout << e.get_coeffvalue( it ) << ", ";
  }
  std::cout << std::endl;
  */
}

void
nest::iaf_psc_delta_eprop::handle( DataLoggingRequest& e )
{
  B_.logger_.handle( e );
}

} // namespace