#include "NullMemoryCommand.hpp"

namespace asmc = axis::services::memory::commands;

asmc::NullMemoryCommand::NullMemoryCommand(void)
{
  // nothing to do here
}

asmc::NullMemoryCommand::~NullMemoryCommand(void)
{
  // nothing to do here
}

void asmc::NullMemoryCommand::Execute( void *, void * )
{
  // nothing to do here
}

asmc::MemoryCommand& asmc::NullMemoryCommand::Clone( void ) const
{
  return *new NullMemoryCommand();
}
