#include "MessageFilter.hpp"
#include "DefaultMessageFilter.hpp"

const axis::services::messaging::filters::MessageFilter& axis::services::messaging::filters::MessageFilter::Default = *new DefaultMessageFilter();

axis::services::messaging::filters::MessageFilter::~MessageFilter( void )
{

}