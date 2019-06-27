/*
 *  eprop_archiving_node.cpp
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

#include "eprop_archiving_node.h"

// Includes from nestkernel:
#include "kernel_manager.h"

// Includes from sli:
#include "dictutils.h"

namespace nest
{

// member functions for Eprop_Archiving_Node

nest::Eprop_Archiving_Node::Eprop_Archiving_Node()
  : Archiving_Node()
  , theta_plus_( -45.3 )
{
}

nest::Eprop_Archiving_Node::Eprop_Archiving_Node(
  const Eprop_Archiving_Node& n )
  : Archiving_Node( n )
  , theta_plus_( n.theta_plus_ )
{
}

void
nest::Eprop_Archiving_Node::get_status( DictionaryDatum& d ) const
{
  Archiving_Node::get_status( d );

  def< double >( d, names::theta_plus, theta_plus_ );
}

void
nest::Eprop_Archiving_Node::set_status( const DictionaryDatum& d )
{
  Archiving_Node::set_status( d );

  // We need to preserve values in case invalid values are set
  double new_theta_plus = theta_plus_;
  updateValue< double >( d, names::theta_plus, new_theta_plus );

  theta_plus_ = new_theta_plus;
}

void
nest::Eprop_Archiving_Node::get_eprop_history( double t1,
  double t2,
  std::deque< histentry_cl >::iterator* start,
  std::deque< histentry_cl >::iterator* finish )
{
  /*
  std::cout << "read hist from " << t1 << " to " << t2 << std::endl;
  std::cout << "whole history: ";
  for ( std::deque< histentry_cl >::iterator itr = eprop_history_.begin();
      itr != eprop_history_.end(); itr++ )
  {
    std::cout << itr->dw_ << ", ";
  }
  std::cout << std::endl;
  */
  *finish = eprop_history_.end();
  if ( eprop_history_.empty() )
  {
    *start = *finish;
    return;
  }
  else
  {
    std::deque< histentry_cl >::iterator runner = eprop_history_.begin();
    // To have a well defined discretization of the integral, we make sure
    // that we exclude the entry at t1 but include the one at t2 by subtracting
    // a small number so that runner->t_ is never equal to t1 or t2.
    while ( ( runner != eprop_history_.end() ) && ( runner->t_ - 1.0e-6 < t1 ) )
    {
      ++runner;
    }
    *start = runner;
    while ( ( runner != eprop_history_.end() ) && ( runner->t_ - 1.0e-6 < t2 ) )
    {
      ( runner->access_counter_ )++;
      ++runner;
    }
    *finish = runner;
  }
}

void
nest::Eprop_Archiving_Node::write_eprop_history( Time const& t_sp,
  double learning_signal )
{
  const double t_ms = t_sp.get_ms();

  if ( n_incoming_ )
  {
    // prune all entries from history which are no longer needed
    // except the penultimate one. we might still need it.
    while ( eprop_history_.size() > 1 )
    {
      if ( eprop_history_.front().access_counter_ >= n_incoming_ )
      {
        eprop_history_.pop_front();
      }
      else
      {
        break;
      }
    }
    // create new entry in history
    eprop_history_.push_back( histentry_cl( t_ms, learning_signal, 0 ) );
  }
}

void
nest::Eprop_Archiving_Node::add_learning_to_hist( DelayedRateConnectionEvent& e )
{
  const double weight = e.get_weight();
  const long delay = e.get_delay_steps();
  const Time stamp = e.get_stamp();

  // TODO: Do we need to sutract the resolution? Examine delays in the network.
  double t_ms = stamp.get_ms() - Time::get_resolution().get_ms();

  std::deque< histentry_cl >::iterator start;
  std::deque< histentry_cl >::iterator finish;
  
  // Get part of history to which the learning signal is added
  // This increases the access counter which is undone below
  nest::Eprop_Archiving_Node::get_eprop_history(
     t_ms, t_ms + Time::delay_steps_to_ms(delay), &start, &finish );

  std::vector< unsigned int >::iterator it = e.begin();

  //std::cout << "t_ms = " << t_ms << std::endl;
  //std::cout << "learning to hist: " << std::endl;

  // The call to get_coeffvalue( it ) in this loop also advances the iterator it
  while ( start != finish && it != e.end() )
  {
    //std::cout << start->t_ << ", ";
    // Add learning signal and reduce access counter
    start->dw_ = weight * ( e.get_coeffvalue( it ) + start->dw_ );
    ( start->access_counter_ )--;
    start++;
  }
  //std::cout << std::endl;
}

} // of namespace nest
