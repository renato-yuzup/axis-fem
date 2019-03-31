#include "stdafx.h"
#include "SystemDateTimeFacet.hpp"
#include <boost/locale.hpp>
#include <iosfwd>

using namespace boost::locale;
using namespace axis::foundation::date_time;

#ifdef _UNICODE
	typedef std::wstringstream string_stream;
#else
	typedef std::stringstream string_stream;
#endif

axis::services::locales::SystemDateTimeFacet::SystemDateTimeFacet( void )
{
// 	generator gen;
// 	_systemLocale = gen("");
}

axis::String axis::services::locales::SystemDateTimeFacet::DoToShortDateString( const axis::foundation::date_time::Date& date ) const
{
	date_time bdate;
	bdate.set(period::period_type(period::marks::day), date.GetDay());
	bdate.set(period::period_type(period::marks::month), date.GetMonth());
	bdate.set(period::period_type(period::marks::year), date.GetYear());
	string_stream ss;
	ss.imbue(_systemLocale);
	ss << as::date_short << bdate;
	return axis::String(ss.str().c_str());
}

axis::String axis::services::locales::SystemDateTimeFacet::DoToLongDateString( const axis::foundation::date_time::Date& date ) const
{
	date_time bdate;
	bdate.set(period::period_type(period::marks::day), date.GetDay());
	bdate.set(period::period_type(period::marks::month), date.GetMonth());
	bdate.set(period::period_type(period::marks::year), date.GetYear());
	string_stream ss;
	ss.imbue(_systemLocale);
	ss << as::date_long << bdate;
	return axis::String(ss.str().c_str());
}

axis::String axis::services::locales::SystemDateTimeFacet::DoToShortTimeString( const axis::foundation::date_time::Time& time ) const
{
	date_time btime;
	btime.set(period::period_type(period::marks::hour), time.GetHours());
	btime.set(period::period_type(period::marks::minute), time.GetMinutes());
	btime.set(period::period_type(period::marks::second), time.GetSeconds());
	string_stream ss;
	ss.imbue(_systemLocale);
	ss << as::time_short << btime;
	return axis::String(ss.str().c_str());
}

axis::String axis::services::locales::SystemDateTimeFacet::DoToLongTimeString( const axis::foundation::date_time::Time& time ) const
{
	date_time btime;
	btime.set(period::period_type(period::marks::hour), time.GetHours());
	btime.set(period::period_type(period::marks::minute), time.GetMinutes());
	btime.set(period::period_type(period::marks::second), time.GetSeconds());
	string_stream ss;
	ss.imbue(_systemLocale);
	ss << as::time_long << btime;
	return axis::String(ss.str().c_str());
}

axis::String axis::services::locales::SystemDateTimeFacet::DoToLongTimeMillisString( const axis::foundation::date_time::Time& time ) const
{
	// milliseconds definition does not appear in locale declarations, so we simply ignore it
	return DoToLongTimeString(time);
}

axis::String axis::services::locales::SystemDateTimeFacet::DoToShortTimeRangeString( const axis::foundation::date_time::Timespan& timespan, axis::foundation::date_time::Timespan::TimespanInterval majorRange ) const
{
	date_time_period_set period;
	if (timespan.GetDays() != 0) period.add(date_time_period(period::day(), timespan.GetDays()));
	if (timespan.GetHours() != 0) period.add(date_time_period(period::hour(), timespan.GetHours()));
	if (timespan.GetMinutes() != 0) period.add(date_time_period(period::minute(), timespan.GetMinutes()));
	if (timespan.GetSeconds() != 0) period.add(date_time_period(period::second(), timespan.GetSeconds()));

	string_stream ss;
	ss.imbue(_systemLocale);
	ss << as::date_short << period;
	return axis::String(ss.str().c_str());
}

axis::String axis::services::locales::SystemDateTimeFacet::DoToLongTimeRangeString( const axis::foundation::date_time::Timespan& timespan, axis::foundation::date_time::Timespan::TimespanInterval majorRange ) const
{
	date_time_period_set period;
	if (timespan.GetDays() != 0) period.add(date_time_period(period::day(), timespan.GetDays()));
	if (timespan.GetHours() != 0) period.add(date_time_period(period::hour(), timespan.GetHours()));
	if (timespan.GetMinutes() != 0) period.add(date_time_period(period::minute(), timespan.GetMinutes()));
	if (timespan.GetSeconds() != 0) period.add(date_time_period(period::second(), timespan.GetSeconds()));

	string_stream ss;
	ss.imbue(_systemLocale);
	ss << as::date_long << period;
	return axis::String(ss.str().c_str());
}

axis::String axis::services::locales::SystemDateTimeFacet::DoToShortDateTimeString( const axis::foundation::date_time::Timestamp& timestamp ) const
{
	axis::foundation::date_time::Date date = timestamp.GetDate();
	axis::foundation::date_time::Time time = timestamp.GetTime();
	date_time bdatetime;
	
	bdatetime.set(period::day(), date.GetDay());
	bdatetime.set(period::month(), date.GetMonth());
	bdatetime.set(period::year(), date.GetYear());

	bdatetime.set(period::hour(), time.GetHours());
	bdatetime.set(period::minute(), time.GetMinutes());
	bdatetime.set(period::second(), time.GetSeconds());

	string_stream ss;
	ss.imbue(_systemLocale);
	ss << as::datetime << bdatetime;
	return axis::String(ss.str().c_str());
}

axis::String axis::services::locales::SystemDateTimeFacet::DoToLongDateTimeString( const axis::foundation::date_time::Timestamp& timestamp ) const
{
	// no difference at all
	return DoToShortDateTimeString(timestamp);
}

axis::String axis::services::locales::SystemDateTimeFacet::DoToShortDateTimeMillisString( const axis::foundation::date_time::Timestamp& timestamp ) const
{
	// locale settings simply ignore fractional seconds
	return DoToShortDateTimeString(timestamp);
}

axis::String axis::services::locales::SystemDateTimeFacet::DoToLongDateTimeMillisString( const axis::foundation::date_time::Timestamp& timestamp ) const
{
	// locale settings simply ignore fractional seconds
	return DoToLongDateTimeString(timestamp);
}

axis::foundation::date_time::Timestamp axis::services::locales::SystemDateTimeFacet::DoToLocalTime( const axis::foundation::date_time::Timestamp& timestamp ) const
{
	TIME_ZONE_INFORMATION tzi;
	SYSTEMTIME systime, localtime;

	systime.wDay = timestamp.GetDate().GetDay();
	systime.wMonth = timestamp.GetDate().GetMonth();
	systime.wYear = timestamp.GetDate().GetYear();

	systime.wHour = timestamp.GetTime().GetHours();
	systime.wMinute = timestamp.GetTime().GetMinutes();
	systime.wSecond = timestamp.GetTime().GetSeconds();
	systime.wMilliseconds = timestamp.GetTime().GetMilliseconds();

	GetTimeZoneInformationForYear(timestamp.GetDate().GetYear(), NULL, &tzi);
	SystemTimeToTzSpecificLocalTime(&tzi, &systime, &localtime);

	return Timestamp(Date(localtime.wYear, localtime.wMonth, localtime.wDay),
					 Time(localtime.wHour, localtime.wMinute, localtime.wSecond, localtime.wMilliseconds));
}

