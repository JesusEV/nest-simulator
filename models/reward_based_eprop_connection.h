/*
 *  reawrd_based_reward_based_eprop_synapse.h
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

#ifndef REWARD_BASED_reward_based_EPROP_SYNAPSE_H
#define REWARD_BASED_reward_based_EPROP_SYNAPSE_H

// C++ includes:
#include <cmath>
#include <typeinfo>

// Includes from nestkernel:
#include "common_synapse_properties.h"
#include "connection.h"
#include "connector_model.h"
#include "event.h"
#include "ring_buffer.h"

// Includes from sli:
#include "dictdatum.h"
#include "dictutils.h"

namespace nest
{

/** @BeginDocumentation
Name: reward_based_eprop_synapse - Synapse type for ... TODO

Description:

clopath_synapse is a connector to create Clopath synapses as defined
in [1]. In contrast to usual STDP, the change of the synaptic weight does
not only depend on the pre- and postsynaptic spike timing but also on the
postsynaptic membrane potential.

Clopath synapses require archiving of continuous quantities. Therefore Clopath
synapses can only be connected to neuron models that are capable of doing this
archiving. So far, compatible models are aeif_psc_delta_clopath and
hh_psc_alpha_clopath.

Parameters:

tau_x    double - Time constant of the trace of the presynaptic spike train.
Wmax     double - Maximum allowed weight.
Wmin     double - Minimum allowed weight.

Other parameters like the amplitudes for depression and facilitation are
stored in in the neuron models that are compatible with the Clopath synapse.

Transmits: SpikeEvent

References:

[1] Clopath et al. (2010) Connectivity reflects coding:
    a model of voltage-based STDP with homeostasis.
    Nature Neuroscience 13:3, 344--352
[2] Clopath and Gerstner (2010) Voltage and spike timing interact
    in STDP – a unified model. Front. Synaptic Neurosci. 2:25
    doi: 10.3389/fnsyn.2010.00025
[3] Voltage-based STDP synapse (Clopath et al. 2010) on ModelDB
    https://senselab.med.yale.edu/ModelDB/showmodel.cshtml?model=144566

Authors: Jonas Stapmanns, David Dahmen, Jan Hahne

SeeAlso: stdp_synapse, aeif_psc_delta_clopath, hh_psc_alpha_clopath
*/
// connections are templates of target identifier type (used for pointer /
// target index addressing) derived from generic connection template

/**
 * Class containing the common properties for all synapses of type eprop
 * connection.
 */
class RewardBasedEpropCommonProperties : public CommonSynapseProperties
{
public:
  /**
   * Default constructor.
   * Sets all property values to defaults.
   */
  RewardBasedEpropCommonProperties();

  /**
   * Get all properties and put them into a dictionary.
   */
  void get_status( DictionaryDatum& d ) const;

  /**
   * Set properties from the values given in dictionary.
   */
  void set_status( const DictionaryDatum& d, ConnectorModel& cm );

  long batch_size_;
  double recall_duration_;
  double learning_rate_; 
  double target_firing_rate_;
  double rate_reg_;
  bool use_adam_; 
  double beta1_adam_; 
  double beta2_adam_; 
  double epsilon_adam_; 

  //-----------------------------
  double rb_gamma_; 
  long move_interval_;
};

RewardBasedEpropCommonProperties::RewardBasedEpropCommonProperties()
  : CommonSynapseProperties()
  , learning_rate_( 0.0001 )
  , rate_reg_( 0. )
  , target_firing_rate_( 10. )
  , batch_size_( 1. )
  , beta1_adam_( 0.9 )
  , beta2_adam_( 0.999 )
  , epsilon_adam_( 1.0e-8 )
  , recall_duration_( 1. )  // in ms
  , use_adam_( false )
  , rb_gamma_( 1.0 )
  , move_interval_( 10 )
{
}

void
RewardBasedEpropCommonProperties::get_status( DictionaryDatum& d ) const
{
  CommonSynapseProperties::get_status( d );
  def< double >( d, names::learning_rate, learning_rate_ );
  def< double >( d, names::rate_reg, rate_reg_ );
  def< double >( d, names::target_firing_rate, target_firing_rate_ );
  def< double >( d, names::batch_size, batch_size_);
  def< double >( d, names::beta1_adam, beta1_adam_);
  def< double >( d, names::beta2_adam, beta2_adam_);
  def< double >( d, names::epsilon_adam, epsilon_adam_);
  def< double >( d, names::recall_duration, recall_duration_);
  def< bool >( d, names::use_adam, use_adam_);
  def< double >( d, names::rb_gamma, rb_gamma_);
  def< long >( d, names::move_interval, move_interval_);
}

void
RewardBasedEpropCommonProperties::set_status( const DictionaryDatum& d, ConnectorModel& cm )
{
  CommonSynapseProperties::set_status( d, cm );
  updateValue< double >( d, names::learning_rate, learning_rate_ );
  updateValue< double >( d, names::rate_reg, rate_reg_ );
  updateValue< double >( d, names::target_firing_rate, target_firing_rate_ );
  updateValue< double >( d, names::batch_size, batch_size_);
  updateValue< double >( d, names::beta1_adam, beta1_adam_);
  updateValue< double >( d, names::beta2_adam, beta2_adam_);
  updateValue< double >( d, names::epsilon_adam, epsilon_adam_);
  updateValue< double >( d, names::recall_duration, recall_duration_);
  updateValue< bool >( d, names::use_adam, use_adam_);

  updateValue< double >( d, names::rb_gamma, recall_duration_);
  updateValue< long >( d, names::move_interval, move_interval_);
}

template < typename targetidentifierT >
class reward_based_eprop_synapse : public Connection< targetidentifierT >
{

public:
  typedef RewardBasedEpropCommonProperties CommonPropertiesType;
  typedef Connection< targetidentifierT > ConnectionBase;

  /**
   * Default Constructor.
   * Sets default values for all parameters. Needed by GenericConnectorModel.
   */
  reward_based_eprop_synapse();


  /**
   * Copy constructor.
   * Needs to be defined properly in order for GenericConnector to work.
   */
  reward_based_eprop_synapse( const reward_based_eprop_synapse& );

  // Explicitly declare all methods inherited from the dependent base
  // ConnectionBase. This avoids explicit name prefixes in all places these
  // functions are used. Since ConnectionBase depends on the template parameter,
  // they are not automatically found in the base class.
  using ConnectionBase::get_delay_steps;
  using ConnectionBase::get_delay;
  using ConnectionBase::get_rport;
  using ConnectionBase::get_target;

  /**
   * Get all properties of this connection and put them into a dictionary.
   */
  void get_status( DictionaryDatum& d ) const;

  /**
   * Set properties of this connection from the values given in dictionary.
   */
  void set_status( const DictionaryDatum& d, ConnectorModel& cm );

  /**
   * Send an event to the receiver of this connection.
   * \param e The event to send
   * \param cp common properties of all synapses (empty).
   */
  void send( Event& e, thread t, const RewardBasedEpropCommonProperties& cp );

  //void optimize( int learning_period_counter_, int &last_learning_period_, const RewardBasedEpropCommonProperties& cp);
  //void aggregate_grads_and_optimize(double grad, int extra_null_grads, double learning_period_counter_);
  void optimize(double learning_period_counter_, const RewardBasedEpropCommonProperties& cp);
  void aggregate_grads_and_optimize(double grad, int extra_null_grads,
          double learning_period_counter_, const RewardBasedEpropCommonProperties& cp, int tid, int sid);

  class ConnTestDummyNode : public ConnTestDummyNodeBase
  {
  public:
    // Ensure proper overriding of overloaded virtual functions.
    // Return values from functions are ignored.
    using ConnTestDummyNodeBase::handles_test_event;
    port
    handles_test_event( SpikeEvent&, rport )
    {
      return invalid_port;
    }
    port
    handles_test_event( DSSpikeEvent&, rport )
    {
      return invalid_port;
    }

  };

  void
  check_connection( Node& s,
    Node& t,
    rport receptor_type,
    const CommonPropertiesType& )
  {
    ConnTestDummyNode dummy_target;

    ConnectionBase::check_connection_( dummy_target, s, t, receptor_type );

    if ( EpropArchivingNode* t_eprop = dynamic_cast< EpropArchivingNode* >( &t ) )
    {
      if (t_eprop->is_eprop_actor() || t_eprop->is_eprop_critic() )  // if target is a readout neuron
      {
        t_eprop->init_eprop_buffers( 3.0 * get_delay() );
      }
      else
      {
        t_eprop->init_eprop_buffers( 2.0 * get_delay() );
      }
    }
    t.register_stdp_connection( t_lastspike_ - get_delay(), get_delay() );
  }

  void
  set_weight( double w )
  {
    weight_ = w;
  }

private:
  // data members of each connection
  double weight_;
  double Wmin_;
  double Wmax_;

  double t_lastspike_;
  double t_lastupdate_;
  int last_learning_period_;
  double keep_traces_;
  double tau_low_pass_e_tr_; // time constant for low pass filtering of the eligibility trace
  double propagator_low_pass_; // exp( -dt / tau_low_pass_e_tr_ )
  // TODO: Find a more efficient way to deal with a recall task, i.e. store eprop history only
  // within recall period. This leads to a discontinuous history which needs a different
  // implementation of get_eprop_history, i.e. binary search.

  std::vector< double > pre_syn_spike_times_;
  double m_adam_;  // auxiliary variable for adam optimizer
  double v_adam_;  // auxiliary variable for adam optimizer
  std::vector< double > grads_;  // vector that stores the gradients of one batch
                                 
  //---------------------------------------------
  double t_current_batch_elem_;                                
  double lpc_;
  std::vector< double > pre_syn_spike_times_extra_;
};


/**
 * Send an event to the receiver of this connection.
 * \param e The event to send
 * \param t The thread on which this connection is stored.
 * \param cp Common properties object, containing the stdp parameters.
 */
template < typename targetidentifierT >
inline void
reward_based_eprop_synapse< targetidentifierT >::send( Event& e,
  thread t,
  const RewardBasedEpropCommonProperties& cp)
{
  double t_spike = e.get_stamp().get_ms();
  int t_mod_T = Time( Time::ms( t_spike-2 ) ).get_steps() % Time( Time::ms( cp.move_interval_ ) ).get_steps();
  // use accessor functions (inherited from Connection< >) to obtain delay and
  // target
  Node* target = get_target( t );
  double dendritic_delay = get_delay();

  std::deque< double > t_batch_history_ = kernel().simulation_manager.get_t_batch_history();

  double t_batch_elem = 0;
  int ii = 0;
  for (auto t: t_batch_history_)
  {
      t_batch_elem = t;
      ii++;
      if (t_batch_elem > t_current_batch_elem_) break;
  }
  int extra_null_grads = t_batch_history_.size() - ii; 
      
  // store times of incoming spikes to enable computation of eligibility trace
  if ( t_batch_elem > t_current_batch_elem_ &&  t_mod_T < 5)
  {
    pre_syn_spike_times_extra_.push_back(t_spike);
  }
  else
  {
    pre_syn_spike_times_.push_back( t_spike );
  }

  if ( t_batch_elem > t_current_batch_elem_ &&  t_mod_T >= 5)
  {
    
    // do update only if this is the first spike in a new inverval T
    if ( 1 )
    {
      // retrive time step of simulation
      double const dt = Time::get_resolution().get_ms();
      // get spike history in relevant range (t1, t2] from post-synaptic neuron
      std::deque< histentry_rbeprop >::iterator start;
      std::deque< histentry_rbeprop >::iterator finish;

      double grad = 0.0;
      if (target->is_eprop_actor() )  // if target is a readout neuron
      {
        pre_syn_spike_times_.insert( --pre_syn_spike_times_.end(), t_batch_elem );
        // set pointers start and finish at the beginning/end of the history of the postsynaptic
        // neuron. The history before the first presyn spike time is not relevant because there
        // z_hat, and therefore the eligibility trace, is zero.
        // Therefore, we use the time of the first presyn spike to indicate where start should
        // be assiged to. finish should point to the last entry of the current update interval.
        target->get_eprop_history(
            pre_syn_spike_times_[0] + 4,            // time for start
            t_batch_elem + dendritic_delay,   // time for finish
            t_current_batch_elem_,  // used to register this update
            t_batch_elem,      // used to register this update
            &start,
            &finish );

        // Compute intervals between two consecutive presynaptic spikes which simplifies the
        // cumputation of the trace z_hat of the presynaptic spikes because z_hat jumps by
        // (1 - propagator_low_pass) at each presyn spike and decays exponentially in between.
        std::vector< double > pre_syn_spk_diff(pre_syn_spike_times_.size() - 1);
        std::adjacent_difference(pre_syn_spike_times_.begin(), --pre_syn_spike_times_.end(),
            pre_syn_spk_diff.begin());
        pre_syn_spk_diff.erase( pre_syn_spk_diff.begin() );

        double last_z_hat = 0.0;
        double last_le_hat = 0.0;

        for ( auto pre_syn_spk_t : pre_syn_spk_diff )
        {
          // jump of z_hat
          last_z_hat += ( 1.0 - propagator_low_pass_ );
          for (int t = 0; t < pre_syn_spk_t; ++t)
          {
            last_le_hat += start->learning_signal_ * last_z_hat;
            double dgrad = start->temporal_diff_error_ * last_le_hat + start->entropy_reg_factor_ * last_z_hat;
            grad += dgrad;  // LSDEBUG!

            // exponential decay of z_hat
            last_z_hat *= propagator_low_pass_;
            last_le_hat *= cp.rb_gamma_;
    
            int last_start_t_ = int(start->t_);
            ++start;
            if ( int(start->t_) != last_start_t_ + 1 ) break;
         }
       }
        grad *= dt;
      }
      else if (target->is_eprop_critic() )  // if target is a readout neuron
      {
        pre_syn_spike_times_.insert( --pre_syn_spike_times_.end(), t_batch_elem );

        // set pointers start and finish at the beginning/end of the history of the postsynaptic
        // neuron. The history before the first presyn spike time is not relevant because there
        // z_hat, and therefore the eligibility trace, is zero.
        // Therefore, we use the time of the first presyn spike to indicate where start should
        // be assiged to. finish should point to the last entry of the current update interval.
        target->get_eprop_history(
            pre_syn_spike_times_[0] + 3,            // time for start
            t_batch_elem + dendritic_delay,   // time for finish
            t_current_batch_elem_,  // used to register this update
            t_batch_elem,      // used to register this update
            &start,
            &finish );

        // Compute intervals between two consecutive presynaptic spikes which simplifies the
        // cumputation of the trace z_hat of the presynaptic spikes because z_hat jumps by
        // (1 - propagator_low_pass) at each presyn spike and decays exponentially in between.
        std::vector< double > pre_syn_spk_diff(pre_syn_spike_times_.size() - 1);
        std::adjacent_difference(pre_syn_spike_times_.begin(), --pre_syn_spike_times_.end(),
            pre_syn_spk_diff.begin());
        pre_syn_spk_diff.erase( pre_syn_spk_diff.begin() );

        double last_z_hat = 0.0;
        double last_le_hat = 0.0;
        for ( auto pre_syn_spk_t : pre_syn_spk_diff )
        {
          // jump of z_hat
          last_z_hat += ( 1.0 - propagator_low_pass_ );
          for (int t = 0; t < pre_syn_spk_t; ++t)
          {
            last_le_hat += start->learning_signal_ * last_z_hat;
            double dgrad = start->temporal_diff_error_ * last_le_hat;
            grad -= dgrad; 
            // exponential decay of z_hat
            last_z_hat *= propagator_low_pass_;
            last_le_hat *= cp.rb_gamma_;

             ++start;
           }
         }
        grad *= dt;
      }
      else  // if target is a neuron of the recurrent network
      {
        pre_syn_spike_times_.insert( --pre_syn_spike_times_.end(), t_batch_elem - dendritic_delay );
        // set pointers start and finish at the beginning/end of the history of the postsynaptic
        // neuron. The history before the first presyn spike time is not relevant because there
        // z_hat, and therefore the eligibility trace, is zero.
        // Therefore, we use the time of the first presyn spike to indicate where start should
        // be assiged to. finish should point to the last entry of the current update interval.
        target->get_eprop_history(
            pre_syn_spike_times_[0] + dendritic_delay,  // time for start
            t_batch_elem,           // time for finish
            t_current_batch_elem_,    // used to register this update
            t_batch_elem,        // used to register this update
            &start,
            &finish );

        double alpha = target->get_leak_propagator();
        double sum_t_prime_new = 0.0;
        // compute the sum of the elegibility trace because it is used for the firing rate
        // regularization
        double sum_elig_tr = 0.0;
        // Compute intervals between two consecutive presynaptic spikes which simplifies the
        // cumputation of the trace z_hat of the presynaptic spikes because z_hat jumps by 1
        // at each presyn spike and decays exponentially in between.
        std::vector< double > pre_syn_spk_diff(pre_syn_spike_times_.size() - 1);
        std::adjacent_difference(pre_syn_spike_times_.begin(), --pre_syn_spike_times_.end(),
            pre_syn_spk_diff.begin());
        // The first entry of pre_syn_spk_diff contains the number of steps between the start
        // of the learning interval and the first presynaptic spike. Since the low-pass filtered
        // presynaptic spike train is zero before the first presynaptic spike, we remove the
        // corresponding entry from pre_syn_spk_diff.
        pre_syn_spk_diff.erase( pre_syn_spk_diff.begin() );
        if ( target->is_eprop_adaptive() )
        {
          // if the target is of type aif_psc_delta_eprop (adaptive threshold)
          double beta = target->get_beta();
          double rho = target->get_adapt_propagator();
          double epsilon = 0.0;
          double last_z_hat = 0.0;
          double last_le_hat = 0.0;
          for ( auto pre_syn_spk_t : pre_syn_spk_diff )
          {
            // jump of z_hat
            last_z_hat += 1.0;
            for (int t = 0; t < pre_syn_spk_t; ++t)
            {
              double pseudo_deriv = start->V_m_;
              double elig_tr = pseudo_deriv * ( last_z_hat  - beta * epsilon );
              sum_elig_tr += elig_tr;
              epsilon = pseudo_deriv * last_z_hat + ( rho - beta * pseudo_deriv ) * epsilon;
              // exponential decay of z_hat
              last_z_hat *= alpha;
              sum_t_prime_new = propagator_low_pass_ * sum_t_prime_new + ( 1.0 -
              propagator_low_pass_ ) * elig_tr;
              last_le_hat += sum_t_prime_new * dt * start->learning_signal_;
              grad += start->temporal_diff_error_ * last_le_hat + start->entropy_reg_factor_ * sum_t_prime_new;
              last_le_hat *= cp.rb_gamma_;
 
              ++start;
            }
          }
        }
        else
        {
          // if the target is of type iaf_psc_delta_eprop
          double last_z_hat = 0.0;
          double last_le_hat = 0.0;

          for ( auto pre_syn_spk_t : pre_syn_spk_diff )
          {
            // jump of z_hat
            last_z_hat += 1.0;
            for (int t = 0; t < pre_syn_spk_t; ++t)
            {
              double pseudo_deriv = start->V_m_;
              double elig_tr = pseudo_deriv * last_z_hat;
              // exponential decay of z_hat
              last_z_hat *= alpha;
              sum_elig_tr += elig_tr;
              sum_t_prime_new = propagator_low_pass_ * sum_t_prime_new + ( 1.0 -
                 propagator_low_pass_ ) * elig_tr;
              last_le_hat += sum_t_prime_new * dt * start->learning_signal_;
              double dgrad = start->temporal_diff_error_ * last_le_hat + start->entropy_reg_factor_ * sum_t_prime_new;
              grad += dgrad;
              last_le_hat *= cp.rb_gamma_;
 
              ++start;
            }
          }
        }
        // TODO: in the evidence accumulation task only the gradients due to the learning signal
        // are divided by the recall duration. We have to investigte how this affects the regression
        // task.
        // divide by the number of recall steps to be compatible with the tf implementation
        grad /= Time( Time::ms( cp.recall_duration_ ) ).get_steps();
        // firing rate regularization
        std::deque< double >::iterator start_spk;
        std::deque< double >::iterator finish_spk;
        target->get_spike_history( t_current_batch_elem_,
            t_batch_elem,
            &start_spk,
            &finish_spk );
        int nspikes = std::distance(start_spk, finish_spk);
        // compute average firing rate since last update. factor 1000 to convert into Hz
        double av_firing_rate = nspikes / (t_batch_elem - t_current_batch_elem_);
        // Eq.(56) TODO: this includes a factor 2.0 which is lacking in the derivation in the
        // manuscript.
        grad += 2. * cp.rate_reg_ * ( av_firing_rate - cp.target_firing_rate_ / 1000.) * sum_elig_tr * dt /
          (t_batch_elem - t_current_batch_elem_);
        // TODO: is the following line needed? (dt = 1.0 anyway)
        grad *= dt;
      }
      // implementation of batches: store all gradients in a vector and compute the weight using
      // their mean value.

      if ( isnan( grad ) )
      {
        std::cout << "gradient is nan; something went terribly wrong!" << std::endl;
      }

      aggregate_grads_and_optimize(grad, extra_null_grads, lpc_, cp, 0, 0);
      t_current_batch_elem_ = t_batch_history_.back();

//      grads_.push_back( grad );
//
//      if ( learning_period_counter_ > last_learning_period_ )
//      {
//        optimize( learning_period_counter_, last_learning_period_, cp );
//      }

      // clear history of presynaptic spike because we don't need them any more
      pre_syn_spike_times_.clear();
      for (auto t: pre_syn_spike_times_extra_)
          pre_syn_spike_times_.push_back(t);
      pre_syn_spike_times_extra_.clear();
      pre_syn_spike_times_.push_back( t_spike );

      // DEBUG: tidy_eprop_history also takes care of the spike_history
      target->tidy_rbeprop_history( t_current_batch_elem_ - dendritic_delay );

    }
  }

  e.set_receiver( *target );
  e.set_weight( weight_ );
  // use accessor functions (inherited from Connection< >) to obtain delay in
  // steps and rport
  e.set_delay_steps( get_delay_steps() );
  e.set_rport( get_rport() );
  e();

  t_lastspike_ = t_spike;
}


template < typename targetidentifierT >
inline void
reward_based_eprop_synapse< targetidentifierT >::aggregate_grads_and_optimize(double grad, 
        int extra_null_grads, double learning_period_counter_, 
        const RewardBasedEpropCommonProperties& cp, int tid, int sid)
{

    grads_.push_back( grad );

    if ( grads_.size() >= cp.batch_size_ )
    {
       lpc_ += 1.0;
       optimize(lpc_, cp);
    }

    for (int i=0; i<extra_null_grads; ++i)
    {
      grads_.push_back( 0.0 );

      if ( grads_.size() >= cp.batch_size_ )
      {
        lpc_ += 1.0;
        optimize(lpc_, cp);
      }
    }
}

template < typename targetidentifierT >
inline void
reward_based_eprop_synapse< targetidentifierT >::optimize(double learning_period_counter_,
        const RewardBasedEpropCommonProperties& cp)
{
        double sum_grads = 0.0;
        for ( auto gr : grads_ )
        {
          sum_grads += gr;
        }

        if (isnan(sum_grads))
        {
            grads_.clear();
            return;
        }

        if ( cp.use_adam_ == 1.0 ) // use adam optimizer
        {
          // divide also by the number of recall steps to be compatible with the tf implementation
          sum_grads /= Time( Time::ms( cp.recall_duration_ ) ).get_steps() * cp.batch_size_;
          m_adam_ = cp.beta1_adam_ * m_adam_ + ( 1.0 - cp.beta1_adam_ ) * sum_grads;
          v_adam_ = cp.beta2_adam_ * v_adam_ + ( 1.0 - cp.beta2_adam_ ) * std::pow( sum_grads, 2 );
          double alpha_t = cp.learning_rate_ * std::sqrt( 1.0 - std::pow( cp.beta2_adam_, learning_period_counter_ ) )
            / ( 1.0 - std::pow( cp.beta1_adam_, learning_period_counter_ ) );
          weight_ -= alpha_t * m_adam_ / ( std::sqrt( v_adam_ ) + cp.epsilon_adam_ );
        }
        else // gradient descent
        {
          // here we do not divide by the number of recall steps (see tf implementation)
          sum_grads /= cp.batch_size_;
          weight_ -= cp.learning_rate_ * sum_grads;
        }
        // check whether the new weight is between Wmin and Wmax
        if ( weight_ > Wmax_ )
        {
          weight_ = Wmax_;
        }
        else if ( weight_ < Wmin_ )
        {
          weight_ = Wmin_;
        }

        // clear the buffer of the gradients so that we can start a new batch
        grads_.clear();
}

template < typename targetidentifierT >
reward_based_eprop_synapse< targetidentifierT >::reward_based_eprop_synapse()
  : ConnectionBase()
  , weight_( 1.0 )
  , Wmin_( 0.0 )
  , Wmax_( 100.0 )
  , t_lastspike_( 0.0 )
  , t_lastupdate_( 0.0 )
  , last_learning_period_( 0 )
  , keep_traces_( true )
  , tau_low_pass_e_tr_( 0.0 )
  , propagator_low_pass_( 0.0 )
  , m_adam_( 0.0 )
  , v_adam_( 0.0 )
  , t_current_batch_elem_( 0.0 )
  , lpc_( 0.0 )
{
}

template < typename targetidentifierT >
reward_based_eprop_synapse< targetidentifierT >::reward_based_eprop_synapse(
  const reward_based_eprop_synapse< targetidentifierT >& rhs )
  : ConnectionBase( rhs )
  , weight_( rhs.weight_ )
  , Wmin_( rhs.Wmin_ )
  , Wmax_( rhs.Wmax_ )
  , t_lastspike_( rhs.t_lastspike_ )
  , t_lastupdate_( rhs.t_lastupdate_ )
  , last_learning_period_( rhs.last_learning_period_ )
  , keep_traces_( rhs.keep_traces_ )
  , tau_low_pass_e_tr_( rhs.tau_low_pass_e_tr_ )
  , propagator_low_pass_( rhs.propagator_low_pass_ )
  , m_adam_( rhs.m_adam_ )
  , v_adam_( rhs.v_adam_ )
  , t_current_batch_elem_( rhs.t_current_batch_elem_ )
  , lpc_( rhs.lpc_ )
{
}

template < typename targetidentifierT >
void
reward_based_eprop_synapse< targetidentifierT >::get_status( DictionaryDatum& d ) const
{
  ConnectionBase::get_status( d );
  def< double >( d, names::weight, weight_ );
  def< double >( d, names::Wmin, Wmin_ );
  def< double >( d, names::Wmax, Wmax_ );
  def< double >( d, names::keep_traces, keep_traces_ );
  def< double >( d, names::tau_decay, tau_low_pass_e_tr_ );
  def< long >( d, names::size_of, sizeof( *this ) );
  def< double >( d, names::m_adam, m_adam_);
  def< double >( d, names::v_adam, v_adam_);
}

template < typename targetidentifierT >
void
reward_based_eprop_synapse< targetidentifierT >::set_status( const DictionaryDatum& d,
  ConnectorModel& cm )
{
  ConnectionBase::set_status( d, cm );
  updateValue< double >( d, names::weight, weight_ );
  updateValue< double >( d, names::Wmin, Wmin_ );
  updateValue< double >( d, names::Wmax, Wmax_ );
  updateValue< double >( d, names::keep_traces, keep_traces_ );
  updateValue< double >( d, names::tau_decay, tau_low_pass_e_tr_ );
  updateValue< double >( d, names::m_adam, m_adam_);
  updateValue< double >( d, names::v_adam, v_adam_);

  const double h = Time::get_resolution().get_ms();
  // TODO: t_nextupdate and t_lastupdate should be initialized even if set_status is not called
  // DEBUG: added + delay to correct for the delay of the learning signal
  // DEBUG II: as RewardBasedEpropCommonProperties is not available here t_nextupdate
  // correct initialization was moved temporarily to the send function. 
  //DEBUG: shifted initial value of t_lastupdate to be in sync with TF code
  t_lastupdate_ = 2.0 * get_delay();
  t_current_batch_elem_ = 2.0;
  lpc_ = 0.0;

  // compute propagator for low pass filtering of eligibility trace
  if ( tau_low_pass_e_tr_ > 0.0 )
  {
    propagator_low_pass_ = exp( -h / tau_low_pass_e_tr_ );
  }
  else if ( tau_low_pass_e_tr_ == 0.0 )
  {
    propagator_low_pass_ = 0.0;
  }
  else
  {
    throw BadProperty( "The synaptic time constant tau_decay must be greater than zero." );
  }

  // check if Wmax >= weight >= Wmin
  /*
  if ( not ( ( Wmax_ >= weight_ ) && ( Wmin_ <= weight_ ) ) )
  {
    throw BadProperty( "Wmax, Wmin and the Weight have to satisfy Wmax >= Weight >= Wmin");
  }
  // check if weight_ and Wmin_ have the same sign
  if ( not ( ( Wmax_ >= 0 && Wmin_ >= 0 ) || ( Wmax_ <= 0 && Wmin_ <= 0 ) ) )
  {
    throw BadProperty( "Weight and Wmin must have same sign." );
  }
  */
}

} // of namespace nest

#endif // of #ifndef REWARD_BASED_reward_based_EPROP_SYNAPSE_H
