#include "BlockProvider.hpp"
#include "foundation/ArgumentException.hpp"
#include <set>

namespace aafp = axis::application::factories::parsers;
namespace aslse = axis::services::language::syntax::evaluation;
namespace asmg = axis::services::management;

class aafp::BlockProvider::Pimpl
{
public:
  typedef std::set<aafp::BlockProvider *> provider_list;
  provider_list providers_;
};

aafp::BlockProvider::BlockProvider( void )
{
	pimpl_ = new Pimpl();
}

aafp::BlockProvider::~BlockProvider(void)
{
	// tells every subprovider to unload
	Pimpl::provider_list::iterator end = pimpl_->providers_.end();
	for (Pimpl::provider_list::iterator it = pimpl_->providers_.begin(); it != end; ++it)
	{
		(**it).OnUnregister(*this);
		// detach from the root module manager
		(**it).UnloadModule(*_rootManager);
	}
	delete pimpl_;
  pimpl_ = nullptr;
}

bool aafp::BlockProvider::IsRegisteredProvider( BlockProvider& provider )
{
	return pimpl_->providers_.find(&provider) == pimpl_->providers_.end();
}

bool aafp::BlockProvider::ContainsProvider( const axis::String& blockName, 
                                            const aslse::ParameterList& paramList ) const
{
	// check with every subprovider if someone can parse this block
  Pimpl::provider_list::iterator end = pimpl_->providers_.end();
  for (Pimpl::provider_list::iterator it = pimpl_->providers_.begin(); it != end; ++it)
  {
    if ((**it).CanParse(blockName, paramList))
    {
      return true;
    }
  }
	// no capable provider found
	return false;
}

aafp::BlockProvider& aafp::BlockProvider::GetProvider( const axis::String& blockName, 
                                                       const aslse::ParameterList& paramList ) const
{
	// check with every subprovider if someone can parse this block
  Pimpl::provider_list::iterator end = pimpl_->providers_.end();
  for (Pimpl::provider_list::iterator it = pimpl_->providers_.begin(); it != end; ++it)
  {
    if ((**it).CanParse(blockName, paramList))
    {	// found it!
      return **it;
    }
  }
	// no capable provider found
	throw axis::foundation::ArgumentException();
}

void aafp::BlockProvider::PostProcessRegistration( asmg::GlobalProviderCatalog& manager )
{
	_rootManager = &manager;
	DoOnPostProcessRegistration(manager);
}

void aafp::BlockProvider::UnloadModule( asmg::GlobalProviderCatalog& manager )
{
	_rootManager = NULL;
	DoOnUnload(manager);
}

void aafp::BlockProvider::DoOnPostProcessRegistration( asmg::GlobalProviderCatalog& rootManager )
{
	// nothing to do in this parsers implementation
}

void aafp::BlockProvider::DoOnUnload( asmg::GlobalProviderCatalog& rootManager )
{
	// nothing to do in this parsers implementation
}

void aafp::BlockProvider::OnRegister( BlockProvider& parent )
{
	// nothing to do in this parsers implementation
}

void aafp::BlockProvider::OnUnregister( BlockProvider& parent )
{
	// nothing to do in this parsers implementation
}

void aafp::BlockProvider::RegisterProvider( BlockProvider& provider )
{
  pimpl_->providers_.insert(&provider);
	provider.OnRegister(*this);
	provider.PostProcessRegistration(*_rootManager);
}

void aafp::BlockProvider::UnregisterProvider( BlockProvider& provider )
{
  pimpl_->providers_.erase(&provider);
	provider.OnUnregister(*this);
	provider.UnloadModule(*_rootManager);
}

bool aafp::BlockProvider::IsLeaf( void )
{
	return false;
}
