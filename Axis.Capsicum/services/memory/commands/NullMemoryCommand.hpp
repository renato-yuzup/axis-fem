#pragma once
#include "MemoryCommand.hpp"

namespace axis { namespace services { namespace memory { namespace commands {

/**
 * Defines a command that does nothing to a memory block.
 */
class NullMemoryCommand : public MemoryCommand
{
public:
  NullMemoryCommand(void);
  ~NullMemoryCommand(void);
  virtual void Execute( void *, void * );
  virtual MemoryCommand& Clone( void ) const;
};

} } } } // namespace axis::services::memory::commands
