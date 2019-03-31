#include "Timestamp.hpp"
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/date_time/local_time/local_time.hpp>
#include <boost/date_time/c_local_time_adjustor.hpp>

using namespace boost::posix_time;
using namespace boost::gregorian;
using namespace boost::local_time;

axis::foundation::date_time::Timestamp::Timestamp( void )
{
	_isLocalTime = true;
}

axis::foundation::date_time::Timestamp::Timestamp( Date date ) : _date(date)
{
	_isLocalTime = true;
}

axis::foundation::date_time::Timestamp::Timestamp( Time time ) : _time(time)
{
	_isLocalTime = true;
}

axis::foundation::date_time::Timestamp::Timestamp( Date date, Time time ) : _date(date), _time(time)
{
	_isLocalTime = true;
}

axis::foundation::date_time::Timestamp::Timestamp( Date date, Time time, bool isUTCTime ) : _date(date), _time(time)
{
	_isLocalTime = !isUTCTime;
}

axis::foundation::date_time::Timestamp::Timestamp( const Timestamp& timestamp ) : 
	_date(timestamp.GetDate()), _time(timestamp.GetTime())
{
	_isLocalTime = timestamp.IsLocalTime();
}

axis::foundation::date_time::Timestamp::~Timestamp( void )
{
	// nothing to do here
}

axis::foundation::date_time::Timestamp axis::foundation::date_time::Timestamp::GetLocalTime( void )
{
	ptime tm = microsec_clock::local_time();
	date dt = tm.date();
	time_duration t = tm.time_of_day();
	return Timestamp(Date(dt.year(), dt.month(), dt.day()), 
					 Time(t.hours(), t.minutes(), t.seconds(), (int)(t.fractional_seconds() * 1000 / time_duration::ticks_per_second())));
}

axis::foundation::date_time::Timestamp axis::foundation::date_time::Timestamp::GetUTCTime( void )
{
	ptime tm = microsec_clock::universal_time();
	date dt = tm.date();
	time_duration t = tm.time_of_day();
	return Timestamp(Date(dt.year(), dt.month(), dt.day()), 
		Time(t.hours(), t.minutes(), t.seconds(), (int)(t.fractional_seconds() * 1000 / time_duration::ticks_per_second())), true);
}

axis::foundation::date_time::Timestamp axis::foundation::date_time::Timestamp::ToUTCTime( void ) const
{
	if (IsUTCTime()) return Timestamp(*this);

	// retrieve local time to get a hold of current timezone
	ptime curLocalTime = microsec_clock::local_time();

	// build custom local time
	date dt(date::year_type(GetDate().GetYear()), date::month_type(GetDate().GetMonth()), date::day_type(GetDate().GetDay()));
	time_duration t(time_duration::hour_type(GetTime().GetHours()), 
					time_duration::min_type(GetTime().GetMinutes()), 
					time_duration::sec_type(GetTime().GetSeconds()), 
					time_duration::fractional_seconds_type(GetTime().GetMilliseconds() * time_duration::ticks_per_second() / 1000));
	ptime myTime(dt, t);
	time_zone_ptr zone(new posix_time_zone(curLocalTime.zone_as_posix_string()));
	local_date_time myLocalTime(myTime, zone);

	// convert to UTC
	ptime utcTime = myLocalTime.utc_time();
	date utcDate = utcTime.date();
	time_duration utcTimeOfDay = utcTime.time_of_day();

	return Timestamp(Date(utcDate.year(), utcDate.month(), utcDate.day()),
					 Time(utcTimeOfDay.hours(), utcTimeOfDay.minutes(), utcTimeOfDay.seconds()), true);
}

axis::foundation::date_time::Timestamp axis::foundation::date_time::Timestamp::ToLocalTime( void ) const
{
	typedef boost::date_time::c_local_adjustor<ptime> local_time_zone;

	if (IsLocalTime()) return Timestamp(*this);

	ptime utcTime(date(date::year_type(GetDate().GetYear()), 
					   date::month_type(GetDate().GetMonth()), 
					   date::day_type(GetDate().GetDay())),
				  time_duration(time_duration::hour_type(GetTime().GetHours()), 
								time_duration::min_type(GetTime().GetMinutes()), 
								time_duration::sec_type(GetTime().GetSeconds()), 
								time_duration::fractional_seconds_type(GetTime().GetMilliseconds() * time_duration::ticks_per_second() / 1000)));
	
	ptime localTime = local_time_zone::utc_to_local(utcTime);

	date dt = localTime.date();
	time_duration t = localTime.time_of_day();

	return Timestamp(Date(dt.year(), dt.month(), dt.day()), 
					 Time(t.hours(), t.minutes(), t.seconds(), (int)(t.fractional_seconds() * 1000 / time_duration::ticks_per_second()))
					);
}

axis::foundation::date_time::Date axis::foundation::date_time::Timestamp::GetDate( void ) const
{
	return _date;
}

axis::foundation::date_time::Time axis::foundation::date_time::Timestamp::GetTime( void ) const
{
	return _time;
}

bool axis::foundation::date_time::Timestamp::IsLocalTime( void ) const
{
	return _isLocalTime;
}

bool axis::foundation::date_time::Timestamp::IsUTCTime( void ) const
{
	return !_isLocalTime;
}

axis::foundation::date_time::Timestamp& axis::foundation::date_time::Timestamp::operator=( const Timestamp& other )
{
	if (&other == this) return *this;
	_date = other.GetDate();
	_time = other.GetTime();
	_isLocalTime = other.IsLocalTime();

	return *this;
}

axis::foundation::date_time::Timespan axis::foundation::date_time::operator-( const Timestamp& t1, const Timestamp& t2 )
{
	date dt1(date::year_type(t1.GetDate().GetYear()), date::month_type(t1.GetDate().GetMonth()), date::day_type(t1.GetDate().GetDay()));
	time_duration td1(time_duration::hour_type(t1.GetTime().GetHours()), 
					  time_duration::min_type(t1.GetTime().GetMinutes()), 
					  time_duration::sec_type(t1.GetTime().GetSeconds()), 
					  time_duration::fractional_seconds_type(t1.GetTime().GetMilliseconds() * time_duration::ticks_per_second() / 1000));
	ptime tm1(dt1, td1);
	date dt2(date::year_type(t2.GetDate().GetYear()), date::month_type(t2.GetDate().GetMonth()), date::day_type(t2.GetDate().GetDay()));
	time_duration td2(time_duration::hour_type(t2.GetTime().GetHours()), 
					  time_duration::min_type(t2.GetTime().GetMinutes()), 
					  time_duration::sec_type(t2.GetTime().GetSeconds()), 
					  time_duration::fractional_seconds_type(t2.GetTime().GetMilliseconds() * time_duration::ticks_per_second() / 1000));
	ptime tm2(dt2, td2);

	time_duration diff = tm1 - tm2;
	return Timespan(diff.hours(), diff.minutes(), diff.seconds(), (long)(diff.fractional_seconds() * 1000 / time_duration::ticks_per_second()));
}

axis::foundation::date_time::Timestamp axis::foundation::date_time::operator+( const Timestamp& time, const Timespan& range )
{
	date dt(date::year_type(time.GetDate().GetYear()), date::month_type(time.GetDate().GetMonth()), date::day_type(time.GetDate().GetDay()));
	time_duration t(time_duration::hour_type(time.GetTime().GetHours()), 
					time_duration::min_type(time.GetTime().GetMinutes()), 
					time_duration::sec_type(time.GetTime().GetSeconds()), 
					time_duration::fractional_seconds_type(time.GetTime().GetMilliseconds() * time_duration::ticks_per_second() / 1000));
	ptime tm(dt, t);

	time_duration interval(time_duration::hour_type(range.GetTotalHours()), 
		time_duration::min_type(range.GetMinutes()), 
		time_duration::sec_type(range.GetSeconds()), 
		time_duration::fractional_seconds_type(range.GetMilliseconds() * time_duration::ticks_per_second() / 1000));

	ptime newTime = tm + interval;
	date newDate = newTime.date();
	time_duration newTimeOfDay = newTime.time_of_day();

	return Timestamp(Date(newDate.year(), newDate.month(), newDate.day()),
					 Time(newTimeOfDay.hours(), newTimeOfDay.minutes(), newTimeOfDay.seconds(),
						  (int)(newTimeOfDay.fractional_seconds() * 1000 / time_duration::ticks_per_second())));
}

axis::foundation::date_time::Timestamp axis::foundation::date_time::operator+( const Timespan& range, const Timestamp& time )
{
	return time + range;
}

axis::foundation::date_time::Timestamp axis::foundation::date_time::operator-( const Timestamp& time, const Timespan& range )
{
	date dt(date::year_type(time.GetDate().GetYear()), date::month_type(time.GetDate().GetMonth()), date::day_type(time.GetDate().GetDay()));
	time_duration t(time_duration::hour_type(time.GetTime().GetHours()), 
		time_duration::min_type(time.GetTime().GetMinutes()), 
		time_duration::sec_type(time.GetTime().GetSeconds()), 
		time_duration::fractional_seconds_type(time.GetTime().GetMilliseconds() * time_duration::ticks_per_second() / 1000));
	ptime tm(dt, t);

	time_duration interval(time_duration::hour_type(range.GetTotalHours()), 
		time_duration::min_type(range.GetMinutes()), 
		time_duration::sec_type(range.GetSeconds()), 
		time_duration::fractional_seconds_type(range.GetMilliseconds() * time_duration::ticks_per_second() / 1000));

	ptime newTime = tm - interval;
	date newDate = newTime.date();
	time_duration newTimeOfDay = newTime.time_of_day();

	return Timestamp(Date(newDate.year(), newDate.month(), newDate.day()),
		Time(newTimeOfDay.hours(), newTimeOfDay.minutes(), newTimeOfDay.seconds(),
		(int)(newTimeOfDay.fractional_seconds() * 1000 / time_duration::ticks_per_second())));
}

