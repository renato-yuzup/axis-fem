#include "EntityCollector.hpp"

namespace aaoc = axis::application::output::collectors;

aaoc::EntityCollector::~EntityCollector( void )
{
  // nothing to do here
}

void axis::application::output::collectors::EntityCollector::Prepare( void )
{
  // base implementation does nothing
}

void axis::application::output::collectors::EntityCollector::TearDown( void )
{
  // base implementation does nothing
}