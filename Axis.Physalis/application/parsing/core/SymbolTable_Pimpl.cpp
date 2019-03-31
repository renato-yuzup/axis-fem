#include "SymbolTable_Pimpl.hpp"
#include <assert.h>
#include "foundation/InvalidOperationException.hpp"

using axis::application::parsing::core::SymbolTable;
using axis::String;

static const long AlreadyDefined = -1;

/****************************************************************************************
*************************** SymbolTable::Pimpl::Symbol members **************************
****************************************************************************************/
SymbolTable::Pimpl::Symbol::Symbol( const axis::String& decoratedName, int symbolId ) :
decoratedName_(decoratedName), symbolId_(symbolId), lastRefreshRound_(0), 
source_(_T("")), location_(_T("")), 
lastRefCount_(0), currentRefCount_(0)
{
  // nothing to do here
}

SymbolTable::Pimpl::Symbol::Symbol( const axis::String& decoratedName, int symbolId, 
                                    const axis::String& sourcePath, 
                                    const axis::String& location ) :
decoratedName_(decoratedName), symbolId_(symbolId), lastRefreshRound_(0), 
source_(sourcePath), location_(location), 
lastRefCount_(0), currentRefCount_(0)
{
  // nothing to do here
}

SymbolTable::Pimpl::Symbol::Symbol( const axis::String& decoratedName, int symbolId, 
                                    const axis::String& sourcePath, 
                                    const axis::String& location, int refreshRound ) :
decoratedName_(decoratedName), symbolId_(symbolId), lastRefreshRound_(refreshRound), 
source_(sourcePath), location_(location), 
lastRefCount_(0), currentRefCount_(0)
{
  // nothing to do here
}

SymbolTable::Pimpl::Symbol::Symbol( const axis::String& decoratedName, int symbolId, 
                                    const axis::String& sourcePath, 
                                    const axis::String& location, 
                                    int refreshRound, long currentRefCount ) :
decoratedName_(decoratedName), symbolId_(symbolId), lastRefreshRound_(refreshRound), 
source_(sourcePath), location_(location), 
lastRefCount_(0), currentRefCount_(currentRefCount)
{
  // nothing to do here
}

SymbolTable::Pimpl::Symbol::Symbol( const axis::String& decoratedName, int symbolId, 
                                    const axis::String& sourcePath, 
                                    const axis::String& location, 
                                    int refreshRound, long currentRefCount, 
                                    long lastRefCount ) :
decoratedName_(decoratedName), symbolId_(symbolId), lastRefreshRound_(refreshRound), 
source_(sourcePath), location_(location), 
lastRefCount_(lastRefCount), currentRefCount_(currentRefCount)
{
  // nothing to do here
}

axis::application::parsing::core::SymbolTable::Pimpl::Symbol::~Symbol( void )
{
  // nothing to do here
}

axis::String SymbolTable::Pimpl::Symbol::GetDecoratedName( void ) const
{
  return decoratedName_;
}

int SymbolTable::Pimpl::Symbol::GetId( void ) const
{
  return symbolId_;
}

axis::String SymbolTable::Pimpl::Symbol::GetSourceString( void ) const
{
  return source_;
}

axis::String SymbolTable::Pimpl::Symbol::GetLocationString( void ) const
{
  return location_;
}

int SymbolTable::Pimpl::Symbol::GetLastRefreshRound( void ) const
{
  return lastRefreshRound_;
}

void SymbolTable::Pimpl::Symbol::SetRefreshRound( int roundIndex )
{
  lastRefreshRound_ = roundIndex;
}

long SymbolTable::Pimpl::Symbol::GetCurrentRefCount( void ) const
{
  return currentRefCount_;
}

void SymbolTable::Pimpl::Symbol::SetCurrentRefCount( long count )
{
  currentRefCount_ = count;
}

void SymbolTable::Pimpl::Symbol::IncrementRefCount( void )
{
  currentRefCount_++;
}

void SymbolTable::Pimpl::Symbol::MarkAsDefined( void )
{
  currentRefCount_ = AlreadyDefined;
}

long SymbolTable::Pimpl::Symbol::GetLastRoundRefCount( void ) const
{
  return currentRefCount_;
}

void SymbolTable::Pimpl::Symbol::SetLastRoundRefCount( long count )
{
  currentRefCount_ = count;
}

void SymbolTable::Pimpl::Symbol::SaveAndResetRefCount( void )
{
  lastRefCount_ = currentRefCount_;
  currentRefCount_ = 0;
}


/****************************************************************************************
****************************** SymbolTable::Pimpl members *******************************
****************************************************************************************/
axis::application::parsing::core::SymbolTable::Pimpl::Pimpl( void )
{
  // nothing to do here
}

axis::application::parsing::core::SymbolTable::Pimpl::~Pimpl( void )
{
  // destroy all symbols
  symbol_map::iterator end = symbols.end();
  for (symbol_map::iterator it = symbols.begin(); it != end; ++it)
  {
    delete it->second;
  }
  symbols.clear();
}

