#include "Timespan.hpp"
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/date_time/gregorian/greg_duration.hpp>
#include <assert.h>

using namespace boost::posix_time;
using namespace boost::gregorian;

class axis::foundation::date_time::Timespan::TimespanData
{
private:
	boost::posix_time::time_duration _timeDuration;
	boost::gregorian::date_duration _dateDuration;
public:
	TimespanData(void)
	{
		// nothing to do here
	}
	TimespanData(const boost::posix_time::time_duration& timeDuration)
	{
		_timeDuration = timeDuration;
	}
	TimespanData(const boost::gregorian::date_duration& dateDuration, const boost::posix_time::time_duration& timeDuration)
	{
		_dateDuration = dateDuration;
		_timeDuration = timeDuration;
	}

	boost::posix_time::time_duration GetTimeSpan(void) const
	{
		return _timeDuration;
	}

	void SetTimeSpan(const boost::posix_time::time_duration& duration)
	{
		_timeDuration = duration;
	}

	boost::gregorian::date_duration GetDateSpan(void) const
	{
		return _dateDuration;
	}

	void SetDateSpan(const boost::gregorian::date_duration& duration)
	{
		_dateDuration = duration;
	}

	long GetDays(void) const
	{
		return _dateDuration.days();
	}

	long GetHours(void) const
	{
		return _timeDuration.hours();
	}

	long GetMinutes(void) const
	{
		return _timeDuration.minutes();
	}
	long GetSeconds(void) const
	{
		return _timeDuration.seconds();
	}
	long GetMilliseconds(void) const
	{
		return (long)(_timeDuration.fractional_seconds() * 1000 / time_duration::ticks_per_second());
	}

};


axis::foundation::date_time::Timespan::Timespan( void )
{
	_data = new TimespanData();
}

axis::foundation::date_time::Timespan::Timespan( const Timespan& timespan )
{
	_data = new TimespanData(timespan._data->GetTimeSpan());
}

axis::foundation::date_time::Timespan::Timespan( long hours, long minutes, long seconds )
{
	time_duration tm(time_duration::hour_type(hours), 
					 time_duration::min_type(minutes),
					 time_duration::sec_type(seconds),
					 time_duration::fractional_seconds_type(0));

	long days = tm.hours() / 24;
	long remainingHours = tm.hours() % 24;

	_data = new TimespanData(date_duration(days), time_duration(
		time_duration::hour_type(remainingHours), 
		time_duration::min_type(minutes),
		time_duration::sec_type(seconds)));
}

axis::foundation::date_time::Timespan::Timespan( long hours, long minutes, long seconds, long milliseconds )
{
	time_duration t(time_duration::hour_type(hours), 
		time_duration::min_type(minutes),
		time_duration::sec_type(seconds),
		time_duration::fractional_seconds_type(milliseconds * (time_duration::ticks_per_second() / 1000)));
	
	long days = t.hours() / 24;
	long remainingHours = t.hours() % 24;

	_data = new TimespanData(date_duration(days),
							 time_duration(time_duration::hour_type(remainingHours), 
							 time_duration::min_type(minutes),
							 time_duration::sec_type(seconds),
							 time_duration::fractional_seconds_type(milliseconds * (time_duration::ticks_per_second() / 1000))));
}

axis::foundation::date_time::Timespan::Timespan( long days, long hours, long minutes, long seconds, long milliseconds )
{
	time_duration t(time_duration::hour_type(hours), 
		time_duration::min_type(minutes),
		time_duration::sec_type(seconds),
		time_duration::fractional_seconds_type(milliseconds * (time_duration::ticks_per_second() / 1000)));

	long totalDays = days + t.hours() / 24;
	long remainingHours = t.hours() % 24;

	_data = new TimespanData(date_duration(totalDays),
							 time_duration(time_duration::hour_type(remainingHours), 
							 time_duration::min_type(minutes),
							 time_duration::sec_type(seconds),
							 time_duration::fractional_seconds_type(milliseconds * (time_duration::ticks_per_second() / 1000))));
}

axis::foundation::date_time::Timespan::Timespan( long value, TimespanInterval intervaltype )
{
	time_duration t(time_duration::hour_type(0), time_duration::min_type(0), time_duration::sec_type(0));
	date_duration d(0);
	switch (intervaltype)
	{
	case axis::foundation::date_time::Timespan::Days:
		d = date_duration(value);
		break;
	case axis::foundation::date_time::Timespan::Hours:
		t = time_duration(time_duration::hour_type(value), time_duration::min_type(0), time_duration::sec_type(0));
		break;
	case axis::foundation::date_time::Timespan::Minutes:
		t = time_duration(time_duration::hour_type(0), time_duration::min_type(value), time_duration::sec_type(0));
		break;
	case axis::foundation::date_time::Timespan::Seconds:
		t = time_duration(time_duration::hour_type(0), time_duration::min_type(0), time_duration::sec_type(value));
		break;
	case axis::foundation::date_time::Timespan::Milliseconds:
		t = time_duration(time_duration::hour_type(0), time_duration::min_type(0), time_duration::sec_type(0), 
						  time_duration::fractional_seconds_type(value * (time_duration::ticks_per_second() / 1000)));
		break;
	default:
		assert(!"Code execution should never reach this line!");
		break;
	}
	long totalDays = d.days() + t.hours() / 24;
	long remainingHours = t.hours() % 24;

	_data = new TimespanData(date_duration(totalDays),
		time_duration(time_duration::hour_type(remainingHours), 
		time_duration::min_type(t.minutes()),
		time_duration::sec_type(t.seconds()),
		time_duration::fractional_seconds_type(t.fractional_seconds())));
}

axis::foundation::date_time::Timespan::Timespan( uint64 tenthsOfMilliseconds )
{
	uint64 totalMillis = tenthsOfMilliseconds / 10;
	uint64 totalSecs   = totalMillis / 1000;
	uint64 totalMins   = totalSecs / 60;
	uint64 totalHours  = totalMins / 60;
	long   days        = (long)(totalHours / 24);

	int millis = (int)(totalMillis % 1000);
	int secs   = (int)(totalSecs % 60);
	int mins   = (int)(totalMins % 60);
	int hours  = (int)(totalHours % 24);

	_data = new TimespanData(date_duration(days),
		time_duration(time_duration::hour_type(hours), 
		time_duration::min_type(mins),
		time_duration::sec_type(secs),
		time_duration::fractional_seconds_type(millis * time_duration::ticks_per_second() / 1000)));
}

axis::foundation::date_time::Timespan::~Timespan( void )
{
	delete _data;
}

long axis::foundation::date_time::Timespan::GetDays( void ) const
{
	return _data->GetDays();
}

long axis::foundation::date_time::Timespan::GetHours( void ) const
{
	return _data->GetHours();
}

long axis::foundation::date_time::Timespan::GetMinutes( void ) const
{
	return _data->GetMinutes();
}

long axis::foundation::date_time::Timespan::GetSeconds( void ) const
{
	return _data->GetSeconds();
}

long axis::foundation::date_time::Timespan::GetMilliseconds( void ) const
{
	return _data->GetMilliseconds();
}

long axis::foundation::date_time::Timespan::GetTotalHours( void ) const
{
	return _data->GetDays() * 24 + _data->GetHours();
}

long axis::foundation::date_time::Timespan::GetTotalMinutes( void ) const
{
	return _data->GetDays() * 1440 + _data->GetTimeSpan().total_seconds() / 60;
}

long axis::foundation::date_time::Timespan::GetTotalSeconds( void ) const
{
	return _data->GetDays() * 8400 + _data->GetTimeSpan().total_seconds();
}

long axis::foundation::date_time::Timespan::GetTotalMilliseconds( void ) const
{
	return _data->GetDays() * 8400000 + _data->GetTimeSpan().total_milliseconds();
}

bool axis::foundation::date_time::Timespan::operator<( const Timespan& other ) const
{
	if (_data->GetDays() < other._data->GetDays()) 
	{
		return true;
	}
	else if (_data->GetDays() == other._data->GetDays())
	{
		return _data->GetTimeSpan() < other._data->GetTimeSpan();
	}

	return false;
}

bool axis::foundation::date_time::Timespan::operator<=( const Timespan& other ) const
{
	if (_data->GetDays() < other._data->GetDays()) 
	{
		return true;
	}
	else if (_data->GetDays() == other._data->GetDays())
	{
		return _data->GetTimeSpan() <= other._data->GetTimeSpan();
	}

	return false;
}

bool axis::foundation::date_time::Timespan::operator>( const Timespan& other ) const
{
	if (_data->GetDays() > other._data->GetDays()) 
	{
		return true;
	}
	else if (_data->GetDays() == other._data->GetDays())
	{
		return _data->GetTimeSpan() > other._data->GetTimeSpan();
	}

	return false;
}

bool axis::foundation::date_time::Timespan::operator>=( const Timespan& other ) const
{
	if (_data->GetDays() > other._data->GetDays()) 
	{
		return true;
	}
	else if (_data->GetDays() == other._data->GetDays())
	{
		return _data->GetTimeSpan() >= other._data->GetTimeSpan();
	}

	return false;
}

bool axis::foundation::date_time::Timespan::operator==( const Timespan& other ) const
{
	return (_data->GetDays() == other._data->GetDays()) &&
		   (_data->GetTimeSpan() == other._data->GetTimeSpan());
}

bool axis::foundation::date_time::Timespan::operator!=( const Timespan& other ) const
{
	return (_data->GetDays() != other._data->GetDays()) ||
		(_data->GetTimeSpan() != other._data->GetTimeSpan());
}

axis::foundation::date_time::Timespan& axis::foundation::date_time::Timespan::operator=( const Timespan& t )
{
	if (&t == this) return *this;
	_data->SetDateSpan(t._data->GetDateSpan());
	_data->SetTimeSpan(t._data->GetTimeSpan());
	return *this;
}

bool axis::foundation::date_time::Timespan::HasFractionalDay( void ) const
{
	return _data->GetTimeSpan().total_milliseconds() != 0;
}

bool axis::foundation::date_time::Timespan::HasWholeDays( void ) const
{
	return _data->GetDays() != 0;
}

axis::foundation::date_time::Timespan axis::foundation::date_time::Timespan::GetWholeDays( void ) const
{
	return Timespan(GetDays(), Days);
}

axis::foundation::date_time::Timespan axis::foundation::date_time::Timespan::GetFractionalDay( void ) const
{
	return Timespan(GetHours(), GetMinutes(), GetSeconds(), GetMilliseconds());
}

axis::foundation::date_time::Timespan axis::foundation::date_time::operator+( const Timespan& t1, const Timespan& t2 )
{
	time_duration t = t1._data->GetTimeSpan() + t2._data->GetTimeSpan();
	long totalDays = t.hours() / 24 + t1.GetDays() + t2.GetDays();
	long totalHours = t.hours() % 24;
	return Timespan(totalDays, totalHours, t.minutes(), t.seconds(), (long)(t.fractional_seconds() * 1000 / time_duration::ticks_per_second()));
}

axis::foundation::date_time::Timespan axis::foundation::date_time::operator-( const Timespan& t1, const Timespan& t2 )
{
	int64 h1, h2;
	h1 = t1.GetTotalHours();
	h2 = t2.GetTotalHours();

	time_duration t = time_duration(time_duration::hour_type(h1), 
									time_duration::min_type(t1.GetMinutes()), 
									time_duration::sec_type(t1.GetSeconds()), 
									time_duration::fractional_seconds_type(t1.GetMilliseconds() * time_duration::ticks_per_second() / 1000)
								   ) -
					  time_duration(time_duration::hour_type(h2), 
								   time_duration::min_type(t2.GetMinutes()), 
								   time_duration::sec_type(t2.GetSeconds()), 
								   time_duration::fractional_seconds_type(t2.GetMilliseconds() * time_duration::ticks_per_second() / 1000)
								   );
	long totalDays = t.hours() / 24;
	long totalHours = t.hours() % 24;
	return Timespan(totalDays, totalHours, t.minutes(), t.seconds(), (long)(t.fractional_seconds() * 1000 / time_duration::ticks_per_second()));
}

axis::foundation::date_time::Timespan axis::foundation::date_time::operator*( const Timespan& t1, int factor )
{
	int64 h1;
	h1 = t1.GetTotalHours();

	time_duration t = time_duration(time_duration::hour_type(h1), 
		time_duration::min_type(t1.GetMinutes()), 
		time_duration::sec_type(t1.GetSeconds()), 
		time_duration::fractional_seconds_type(t1.GetMilliseconds() * time_duration::ticks_per_second() / 1000)
		) * factor;
	long totalDays = t.hours() / 24;
	long totalHours = t.hours() % 24;
	return Timespan(totalDays, totalHours, t.minutes(), t.seconds(), (long)(t.fractional_seconds() * 1000 / time_duration::ticks_per_second()));
}

axis::foundation::date_time::Timespan axis::foundation::date_time::operator*( const Timespan& t1, real factor )
{
  int64 h1;
  h1 = t1.GetTotalHours();

  int integralPart = (int)factor;
  const int relevantDigits = 100000;
  int fractionalPart = (int)((factor - integralPart)*relevantDigits);

  time_duration tIntegral = time_duration(time_duration::hour_type(h1), 
    time_duration::min_type(t1.GetMinutes()), 
    time_duration::sec_type(t1.GetSeconds()), 
    time_duration::fractional_seconds_type(t1.GetMilliseconds() * time_duration::ticks_per_second() / 1000)
    ) * integralPart;

  time_duration tFractional = time_duration(time_duration::hour_type(h1), 
    time_duration::min_type(t1.GetMinutes()), 
    time_duration::sec_type(t1.GetSeconds()), 
    time_duration::fractional_seconds_type(t1.GetMilliseconds() * time_duration::ticks_per_second() / 1000)
    ) * fractionalPart;
  tFractional = tFractional / relevantDigits;

  time_duration t = tIntegral + tFractional;
  long totalDays = t.hours() / 24;
  long totalHours = t.hours() % 24;
  return Timespan(totalDays, totalHours, t.minutes(), t.seconds(), (long)(t.fractional_seconds() * 1000 / time_duration::ticks_per_second()));
}

axis::foundation::date_time::Timespan axis::foundation::date_time::operator*( int factor, const Timespan& t1 )
{
	return t1 * factor;
}

axis::foundation::date_time::Timespan axis::foundation::date_time::operator*( real factor, const Timespan& t1 )
{
  return t1 * factor;
}

axis::foundation::date_time::Timespan axis::foundation::date_time::operator/( const Timespan& t1, int factor )
{
	int64 h1;
	h1 = t1.GetTotalHours();

	time_duration t = time_duration(time_duration::hour_type(h1), 
		time_duration::min_type(t1.GetMinutes()), 
		time_duration::sec_type(t1.GetSeconds()), 
		time_duration::fractional_seconds_type(t1.GetMilliseconds() * time_duration::ticks_per_second() / 1000)
		) / factor;
	long totalDays = t.hours() / 24;
	long totalHours = t.hours() % 24;
	return Timespan(totalDays, totalHours, t.minutes(), t.seconds(), (long)(t.fractional_seconds() * 1000 / time_duration::ticks_per_second()));
}

axis::foundation::date_time::Timespan axis::foundation::date_time::operator/( const Timespan& t1, real factor )
{
  int64 h1;
  h1 = t1.GetTotalHours();

  int integralPart = (int)factor;
  const int relevantDigits = 100000;
  int fractionalPart = (int)((factor - integralPart)*relevantDigits);

  time_duration t = time_duration(time_duration::hour_type(h1), 
    time_duration::min_type(t1.GetMinutes()), 
    time_duration::sec_type(t1.GetSeconds()), 
    time_duration::fractional_seconds_type(t1.GetMilliseconds() * time_duration::ticks_per_second() / 1000)
    ) * relevantDigits;
  t = t / (integralPart*relevantDigits + fractionalPart);
  long totalDays = t.hours() / 24;
  long totalHours = t.hours() % 24;
  return Timespan(totalDays, totalHours, t.minutes(), t.seconds(), (long)(t.fractional_seconds() * 1000 / time_duration::ticks_per_second()));
}
