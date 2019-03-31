#include "Time.hpp"
#include <boost/date_time/posix_time/posix_time.hpp>
#include "foundation/ArgumentException.hpp"
#include "foundation/OutOfRangeException.hpp"

using namespace boost::posix_time;

static const long ticks_per_day = 24 * 3600 * 1000 * (long)(time_duration::ticks_per_second() / 1000);

class axis::foundation::date_time::Time::TimeData
{
private:
	boost::posix_time::time_duration _time;
public:
	TimeData(void) : _time(time_duration::hour_type(0), time_duration::min_type(0), time_duration::sec_type(0))
	{
		// nothing to do here
	}
	boost::posix_time::time_duration GetTime(void) const
	{
		return _time;
	}
	void SetTime(const boost::posix_time::time_duration& time)
	{
		_time = time;
	}
	int GetHours(void) const
	{
		return _time.hours();
	}
	int GetMinutes(void) const
	{
		return _time.minutes();
	}
	int GetSeconds(void) const
	{
		return _time.seconds();
	}
	int GetMilliseconds(void) const
	{
		return (int)(_time.fractional_seconds()*1000 / boost::posix_time::time_duration::ticks_per_second());
	}
};



axis::foundation::date_time::Time::Time( void )
{
	_data = new TimeData();
}

axis::foundation::date_time::Time::Time( const Time& other )
{
	_data = new TimeData();
	_data->SetTime(other._data->GetTime());
}

axis::foundation::date_time::Time::Time( int hours, int minutes, int seconds )
{
	time_duration t(hours, minutes, seconds);
	if (t.total_milliseconds() < 0 || t.total_milliseconds() > ticks_per_day)
	{
		throw axis::foundation::ArgumentException(_T("Invalid time."));
	}
	_data = new TimeData();
	_data->SetTime(t);
}

axis::foundation::date_time::Time::Time( int hours, int minutes, int seconds, int milliseconds )
{
	int millis = (int)(milliseconds * (time_duration::ticks_per_second() / 1000));
	time_duration t(hours, minutes, seconds, millis);
	if (t.total_milliseconds() < 0 || t.total_milliseconds() > ticks_per_day)
	{
		throw axis::foundation::ArgumentException(_T("Invalid time."));
	}
	_data = new TimeData();
	_data->SetTime(t);
}

axis::foundation::date_time::Time::~Time( void )
{
	delete _data;
}

int axis::foundation::date_time::Time::GetHours( void ) const
{
	return _data->GetHours();
}

int axis::foundation::date_time::Time::GetMinutes( void ) const
{
	return _data->GetMinutes();
}

int axis::foundation::date_time::Time::GetSeconds( void ) const
{
	return _data->GetSeconds();
}

int axis::foundation::date_time::Time::GetMilliseconds( void ) const
{
	return _data->GetMilliseconds();
}

long axis::foundation::date_time::Time::GetTotalMinutes( void ) const
{
	return GetHours()*60 + GetMinutes();
}

long axis::foundation::date_time::Time::GetTotalSeconds( void ) const
{
	return GetTotalMinutes()*60 + GetSeconds();
}

long axis::foundation::date_time::Time::GetTotalMilliseconds( void ) const
{
	return GetTotalSeconds()*1000 + GetMilliseconds();
}

bool axis::foundation::date_time::Time::operator<( const Time& other ) const
{
	return _data->GetTime() < other._data->GetTime();
}

bool axis::foundation::date_time::Time::operator<=( const Time& other ) const
{
	return _data->GetTime() <= other._data->GetTime();
}

bool axis::foundation::date_time::Time::operator>( const Time& other ) const
{
	return _data->GetTime() > other._data->GetTime();
}

bool axis::foundation::date_time::Time::operator>=( const Time& other ) const
{
	return _data->GetTime() >= other._data->GetTime();
}

bool axis::foundation::date_time::Time::operator==( const Time& other ) const
{
	return _data->GetTime() == other._data->GetTime();
}

bool axis::foundation::date_time::Time::operator!=( const Time& other ) const
{
	return _data->GetTime() != other._data->GetTime();
}

axis::foundation::date_time::Time& axis::foundation::date_time::Time::operator=( const Time& other )
{
	if (this == &other) return *this;
	_data->SetTime(other._data->GetTime());
	return *this;
}

axis::foundation::date_time::Time axis::foundation::date_time::Time::Now( void )
{
	ptime p = microsec_clock::local_time();
	return Time(p.time_of_day().hours(), p.time_of_day().minutes(), 
				p.time_of_day().seconds(), (int)(p.time_of_day().fractional_seconds() * 1000 / time_duration::ticks_per_second()));
}

axis::foundation::date_time::Time axis::foundation::date_time::operator+( const Time& time, const Timespan& timespan )
{
	if (timespan.HasWholeDays())
	{
		throw axis::foundation::ArgumentException(_T("Cannot operate on whole days."));
	}

	time_duration t1(time.GetHours(), time.GetMinutes(), time.GetSeconds(), time.GetMilliseconds() * time_duration::ticks_per_second() / 1000);
	time_duration t2(timespan.GetHours(), timespan.GetMinutes(), timespan.GetSeconds(), timespan.GetMilliseconds() * time_duration::ticks_per_second() / 1000);

	time_duration t = t1 + t2;
	if (t.total_milliseconds() < 0 || t.total_milliseconds() > ticks_per_day)
	{
		throw axis::foundation::OutOfRangeException(_T("Result is not a valid time."));
	}
	
	return Time(t.hours(), t.minutes(), t.seconds(), (int)(t.fractional_seconds() * 1000 / time_duration::ticks_per_second()));
}

axis::foundation::date_time::Time axis::foundation::date_time::operator-( const Time& time, const Timespan& timespan )
{
	if (timespan.HasWholeDays())
	{
		throw axis::foundation::ArgumentException(_T("Cannot operate on whole days."));
	}

	time_duration t1(time.GetHours(), time.GetMinutes(), time.GetSeconds(), time.GetMilliseconds() * time_duration::ticks_per_second() / 1000);
	time_duration t2(timespan.GetHours(), timespan.GetMinutes(), timespan.GetSeconds(), timespan.GetMilliseconds() * time_duration::ticks_per_second() / 1000);

	time_duration t = t1 - t2;
	if (t.total_milliseconds() < 0 || t.total_milliseconds() > ticks_per_day)
	{
		throw axis::foundation::OutOfRangeException(_T("Result is not a valid time."));
	}

	return Time(t.hours(), t.minutes(), t.seconds(), (int)(t.fractional_seconds() * 1000 / time_duration::ticks_per_second()));
}

axis::foundation::date_time::Timespan axis::foundation::date_time::operator-( const Time& t1, const Time& t2 )
{
	time_duration tm1(t1.GetHours(), t1.GetMinutes(), t1.GetSeconds(), t1.GetMilliseconds() * time_duration::ticks_per_second() / 1000);
	time_duration tm2(t2.GetHours(), t2.GetMinutes(), t2.GetSeconds(), t2.GetMilliseconds() * time_duration::ticks_per_second() / 1000);

	time_duration t = tm1 - tm2;

	return Timespan(t.hours(), t.minutes(), t.seconds(), (long)(t.fractional_seconds() * 1000 / time_duration::ticks_per_second()));
}
