#include "SnapshotMark.hpp"
#include "foundation/ArgumentException.hpp"


axis::domain::analyses::SnapshotMark::SnapshotMark( real time )
{
	if (time < 0)
	{
		throw axis::foundation::ArgumentException(_T("time"));
	}
	_time = time;
}

axis::domain::analyses::SnapshotMark::~SnapshotMark( void )
{
	// nothing to do here
}

void axis::domain::analyses::SnapshotMark::Destroy( void ) const
{
	delete this;
}

real axis::domain::analyses::SnapshotMark::GetTime( void ) const
{
	return _time;
}

axis::domain::analyses::SnapshotMark& axis::domain::analyses::SnapshotMark::Clone( void ) const
{
	return *new SnapshotMark(*this);
}