#include "ParserEngine.hpp"
#include "ParserEngineImpl.hpp"
#include "foundation/InvalidOperationException.hpp"

namespace aaa = axis::application::agents;
namespace aaj = axis::application::jobs;
namespace asmg = axis::services::management;

aaa::ParserEngine::ParserEngine( void )
{
  pimpl_ = NULL;
}

aaa::ParserEngine::~ParserEngine( void )
{
	delete pimpl_;
}

void aaa::ParserEngine::SetUp( asmg::GlobalProviderCatalog& manager )
{
  pimpl_ = new ParserEngineImpl(manager);
}

void aaa::ParserEngine::Parse( aaj::StructuralAnalysis& analysis,
                               const axis::String& masterInputFilename, 
													     const axis::String& baseIncludePath )
{
	if (pimpl_ == NULL)
	{
		throw axis::foundation::InvalidOperationException(_T("Must set up first."));
	}
  pimpl_->ConnectListener(*this);
	pimpl_->Parse(analysis, masterInputFilename, baseIncludePath);
  pimpl_->DisconnectListener(*this);
}

void aaa::ParserEngine::AddPreProcessorSymbol( const axis::String& symbolName )
{
	if (pimpl_ == NULL)
	{
	  throw axis::foundation::InvalidOperationException(_T("Must set up first."));
	}
	pimpl_->AddPreProcessorSymbol(symbolName);
}

void aaa::ParserEngine::ClearPreProcessorSymbols( void )
{
  if (pimpl_ == NULL)
  {
    throw axis::foundation::InvalidOperationException(_T("Must set up first."));
  }
	pimpl_->ClearPreProcessorSymbols();
}
