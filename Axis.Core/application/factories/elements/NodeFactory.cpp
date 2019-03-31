#include "NodeFactory.hpp"
#include "domain/elements/Node.hpp"
#include "services/management/ServiceLocator.hpp"

namespace aafe = axis::application::factories::elements;
namespace ade = axis::domain::elements;
namespace afm = axis::foundation::memory;
namespace asmg = axis::services::management;

aafe::NodeFactory::NodeFactory(void)
{
	_nextId = 0;
}

aafe::NodeFactory::~NodeFactory(void)
{
	// nothing to do
}

afm::RelativePointer aafe::NodeFactory::CreateNode( ade::Node::id_type userId )
{
	return ade::Node::Create(_nextId++, userId);
}

afm::RelativePointer aafe::NodeFactory::CreateNode( ade::Node::id_type userId, 
                                                    coordtype x, coordtype y )
{
	return ade::Node::Create(_nextId++, userId, x, y);
}

afm::RelativePointer aafe::NodeFactory::CreateNode( ade::Node::id_type userId, coordtype x, 
                                                    coordtype y, coordtype z )
{
	return ade::Node::Create(_nextId++, userId, x, y, z);
}

const char * aafe::NodeFactory::GetFeaturePath( void ) const
{
	return axis::services::management::ServiceLocator::GetNodeFactoryPath();
}

const char * aafe::NodeFactory::GetFeatureName( void ) const
{
	return "StandardNodeFactory";
}

void aafe::NodeFactory::UnloadModule( asmg::GlobalProviderCatalog& manager )
{
	// nothing to do here
}

void aafe::NodeFactory::PostProcessRegistration( asmg::GlobalProviderCatalog& manager )
{
	// nothing to do here
}
