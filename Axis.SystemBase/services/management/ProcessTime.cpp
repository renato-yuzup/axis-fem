#include "ProcessTime.hpp"


axis::services::management::ProcessTime::ProcessTime( void )
{
	// nothing to do here
}

axis::services::management::ProcessTime::ProcessTime(	axis::foundation::date_time::Timestamp creationTime, 
														axis::foundation::date_time::Timespan userTime, 
														axis::foundation::date_time::Timespan kernelTime )
{
	_creationTime = creationTime;
	_userTime = userTime;
	_kernelTime = kernelTime;
}

axis::services::management::ProcessTime::~ProcessTime( void )
{
	// nothing to do here
}

axis::foundation::date_time::Timestamp axis::services::management::ProcessTime::CreationTime( void ) const
{
	return _creationTime;
}

axis::foundation::date_time::Timespan axis::services::management::ProcessTime::UserTime( void ) const
{
	return _userTime;
}

axis::foundation::date_time::Timespan axis::services::management::ProcessTime::KernelTime( void ) const
{
	return _kernelTime;
}