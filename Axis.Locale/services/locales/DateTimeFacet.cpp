#include "stdafx.h"
#include "DateTimeFacet.hpp"
#include <assert.h>
#include <math.h>

using namespace axis;
using namespace axis::foundation::date_time;


axis::services::locales::DateTimeFacet::DateTimeFacet( void )
{
	// nothing to do here
}

axis::services::locales::DateTimeFacet::~DateTimeFacet( void )
{
	// nothing to do here
}

axis::String axis::services::locales::DateTimeFacet::DoToShortDateString( const axis::foundation::date_time::Date& date ) const
{
  String s = String::int_parse((long)date.GetDay(), 2) + _T("-") +
 			       String::int_parse((long)date.GetMonth(), 2) + _T("-") +
 			       String::int_parse((long)date.GetYear());  
  return s.replace(_T(" "), _T("0"));
}

axis::String axis::services::locales::DateTimeFacet::DoToLongDateString( const axis::foundation::date_time::Date& date ) const
{
	String monthName;
	int month = date.GetMonth();
	switch (month)
	{
	case 1:  monthName = _T("January");   break;
	case 2:  monthName = _T("Febuary");   break;
	case 3:  monthName = _T("March");     break;
	case 4:  monthName = _T("April");     break;
	case 5:  monthName = _T("May");       break;
	case 6:  monthName = _T("June");      break;
	case 7:  monthName = _T("July");      break;
	case 8:  monthName = _T("August");    break;
	case 9:  monthName = _T("September"); break;
	case 10: monthName = _T("October");   break;
	case 11: monthName = _T("November");  break;
	case 12: monthName = _T("December");  break;
	default:
		assert(!"Code execution should never reach here!");
		break;
	}
	return  String::int_parse((long)date.GetDay()).replace(_T(" "), _T("0")) + _T(" ") +
			monthName + _T(" ") +
			String::int_parse((long)date.GetYear()).replace(_T(" "), _T("0"));
}

axis::String axis::services::locales::DateTimeFacet::DoToShortTimeString( const axis::foundation::date_time::Time& time ) const
{
	return String::int_parse((long)time.GetHours(), 2).replace(_T(" "), _T("0")) + _T(":") +
		   String::int_parse((long)time.GetMinutes(), 2).replace(_T(" "), _T("0"));
}

axis::String axis::services::locales::DateTimeFacet::DoToLongTimeString( const axis::foundation::date_time::Time& time ) const
{
	return String::int_parse((long)time.GetHours(), 2).replace(_T(" "), _T("0")) + _T(":") +
		   String::int_parse((long)time.GetMinutes(), 2).replace(_T(" "), _T("0")) + _T(":") +
		   String::int_parse((long)time.GetSeconds(), 2).replace(_T(" "), _T("0"));
}

axis::String axis::services::locales::DateTimeFacet::DoToLongTimeMillisString( const axis::foundation::date_time::Time& time ) const
{
	return String::int_parse((long)time.GetHours(), 2).replace(_T(" "), _T("0")) + _T(":") +
		   String::int_parse((long)time.GetMinutes(), 2).replace(_T(" "), _T("0")) + _T(":") +
		   String::int_parse((long)time.GetSeconds(), 2).replace(_T(" "), _T("0")) + _T(".") +
		   String::int_parse((long)time.GetMilliseconds(), 3).replace(_T(" "), _T("0"));
}

axis::String axis::services::locales::DateTimeFacet::DoToShortTimeRangeString( const axis::foundation::date_time::Timespan& timespan, axis::foundation::date_time::Timespan::TimespanInterval majorRange ) const
{
	long millis = 0, 
		 secs   = 0, 
		 mins   = 0, 
		 hours  = 0, 
		 days   = 0;

	switch (majorRange)
	{
	case axis::foundation::date_time::Timespan::Days:
		millis = timespan.GetMilliseconds();
		secs   = timespan.GetSeconds();
		mins   = timespan.GetMinutes();
		hours  = timespan.GetHours();
		days   = timespan.GetDays();
		break;
	case axis::foundation::date_time::Timespan::Hours:
		millis = timespan.GetMilliseconds();
		secs   = timespan.GetSeconds();
		mins   = timespan.GetMinutes();
		hours  = timespan.GetTotalHours();
		break;
	case axis::foundation::date_time::Timespan::Minutes:
		millis = timespan.GetMilliseconds();
		secs   = timespan.GetSeconds();
		mins   = timespan.GetTotalMinutes();
		break;
	case axis::foundation::date_time::Timespan::Seconds:
		millis = timespan.GetMilliseconds();
		secs   = timespan.GetTotalSeconds();
		break;
	case axis::foundation::date_time::Timespan::Milliseconds:
		millis = timespan.GetTotalMilliseconds();
		break;
	default:
		assert(!"Code execution should never reach here!");
		break;
	}

	String s;
	if (days == 0 && hours == 0 && mins == 0 && secs == 0 && millis == 0)
	{
		return _T("0min 0.000s");
	}

	if (days != 0) s += String::int_parse(days) + _T("d, ");
	if (hours != 0) s += String::int_parse(hours) + _T("h ");
	if (mins != 0) s += String::int_parse(mins) + _T("m ");
	if (millis != 0)
	{
		s += String::int_parse(secs) + _T(".") + String::int_parse(millis, 3) + _T("s");
	}
	else
	{
		if (secs != 0) s += String::int_parse(secs) + _T("s");
	}

	return s;
}

axis::String axis::services::locales::DateTimeFacet::DoToLongTimeRangeString( const axis::foundation::date_time::Timespan& timespan, axis::foundation::date_time::Timespan::TimespanInterval majorRange ) const
{
	long millis = 0, 
		 secs   = 0, 
		 mins   = 0, 
		 hours  = 0, 
		 days   = 0;

	switch (majorRange)
	{
	case axis::foundation::date_time::Timespan::Days:
		millis = timespan.GetMilliseconds();
		secs   = timespan.GetSeconds();
		mins   = timespan.GetMinutes();
		hours  = timespan.GetHours();
		days   = timespan.GetDays();
		break;
	case axis::foundation::date_time::Timespan::Hours:
		millis = timespan.GetMilliseconds();
		secs   = timespan.GetSeconds();
		mins   = timespan.GetMinutes();
		hours  = timespan.GetTotalHours();
		break;
	case axis::foundation::date_time::Timespan::Minutes:
		millis = timespan.GetMilliseconds();
		secs   = timespan.GetSeconds();
		mins   = timespan.GetTotalMinutes();
		break;
	case axis::foundation::date_time::Timespan::Seconds:
		millis = timespan.GetMilliseconds();
		secs   = timespan.GetTotalSeconds();
		break;
	case axis::foundation::date_time::Timespan::Milliseconds:
		millis = timespan.GetTotalMilliseconds();
		break;
	default:
		assert(!"Code execution should never reach here!");
		break;
	}

	String s;
	if (days == 0 && hours == 0 && mins == 0 && secs == 0 && millis == 0)
	{
		return _T("0 days, 00:00:00.000");
	}

	if (days != 0) s += String::int_parse(days) + (abs(days) == 1? _T(" day, ") : _T(" days, "));
	s += String::int_parse(hours, 2) + _T(":");
	s += String::int_parse(mins, 2) + _T(":");
	if (millis != 0)
	{
		s += String::int_parse(secs, 2) + _T(".") + String::int_parse(millis, 3);
	}
	else
	{
		s += String::int_parse(secs, 2);
	}

	return s;
}

axis::String axis::services::locales::DateTimeFacet::DoToShortDateTimeString( const axis::foundation::date_time::Timestamp& timestamp ) const
{
	return ToShortDateString(timestamp.GetDate()) + _T(" ") + ToShortTimeString(timestamp.GetTime());
}

axis::String axis::services::locales::DateTimeFacet::DoToLongDateTimeString( const axis::foundation::date_time::Timestamp& timestamp ) const
{
	return ToLongDateString(timestamp.GetDate()) + _T(", ") + ToLongTimeString(timestamp.GetTime());
}

axis::String axis::services::locales::DateTimeFacet::DoToShortDateTimeMillisString( const axis::foundation::date_time::Timestamp& timestamp ) const
{
	return ToShortDateString(timestamp.GetDate()) + _T(" ") + ToLongTimeMillisString(timestamp.GetTime());
}

axis::String axis::services::locales::DateTimeFacet::DoToLongDateTimeMillisString( const axis::foundation::date_time::Timestamp& timestamp ) const
{
	return ToLongDateString(timestamp.GetDate()) + _T(", ") + ToLongTimeMillisString(timestamp.GetTime());
}

axis::foundation::date_time::Timestamp axis::services::locales::DateTimeFacet::DoToLocalTime( const axis::foundation::date_time::Timestamp& timestamp ) const
{
	return Timestamp(timestamp.GetDate(), timestamp.GetTime(), false);
}




axis::String axis::services::locales::DateTimeFacet::ToShortDateString( const axis::foundation::date_time::Date& date ) const
{
	String s = DoToShortDateString(date);
	s.trim();
	assert(!s.empty() && "Invalid date formatting.");
	return s;
}

axis::String axis::services::locales::DateTimeFacet::ToLongDateString( const axis::foundation::date_time::Date& date ) const
{
	String s = DoToLongDateString(date);
	s.trim();
	assert(!s.empty() && "Invalid date formatting.");
	return s;
}

axis::String axis::services::locales::DateTimeFacet::ToShortTimeString( const axis::foundation::date_time::Time& time ) const
{
	String s = DoToShortTimeString(time);
	s.trim();
	assert(!s.empty() && "Invalid time formatting.");
	return s;
}

axis::String axis::services::locales::DateTimeFacet::ToLongTimeString( const axis::foundation::date_time::Time& time ) const
{
	String s = DoToLongTimeString(time);
	s.trim();
	assert(!s.empty() && "Invalid time formatting.");
	return s;
}

axis::String axis::services::locales::DateTimeFacet::ToLongTimeMillisString( const axis::foundation::date_time::Time& time ) const
{
	String s = DoToLongTimeMillisString(time);
	s.trim();
	assert(!s.empty() && "Invalid time formatting.");
	return s;
}

axis::String axis::services::locales::DateTimeFacet::ToShortTimeRangeString( const axis::foundation::date_time::Timespan& timespan, axis::foundation::date_time::Timespan::TimespanInterval majorRange /*= axis::foundation::date_time::Timespan::Days*/ ) const
{
	String s = DoToShortTimeRangeString(timespan, majorRange);
	s.trim();
	assert(!s.empty() && "Invalid date formatting.");
	return s;
}

axis::String axis::services::locales::DateTimeFacet::ToLongTimeRangeString( const axis::foundation::date_time::Timespan& timespan, axis::foundation::date_time::Timespan::TimespanInterval majorRange /*= axis::foundation::date_time::Timespan::Days*/ ) const
{
	String s = DoToLongTimeRangeString(timespan, majorRange);
	s.trim();
	assert(!s.empty() && "Invalid date formatting.");
	return s;
}

axis::String axis::services::locales::DateTimeFacet::ToShortDateTimeString( const axis::foundation::date_time::Timestamp& timestamp ) const
{
	String s = DoToShortDateTimeString(timestamp);
	s.trim();
	assert(!s.empty() && "Invalid date formatting.");
	return s;
}

axis::String axis::services::locales::DateTimeFacet::ToLongDateTimeString( const axis::foundation::date_time::Timestamp& timestamp ) const
{
	String s = DoToLongDateTimeString(timestamp);
	s.trim();
	assert(!s.empty() && "Invalid date formatting.");
	return s;
}

axis::String axis::services::locales::DateTimeFacet::ToShortDateTimeMillisString( const axis::foundation::date_time::Timestamp& timestamp ) const
{
	String s = DoToShortDateTimeMillisString(timestamp);
	s.trim();
	assert(!s.empty() && "Invalid date formatting.");
	return s;
}

axis::String axis::services::locales::DateTimeFacet::ToLongDateTimeMillisString( const axis::foundation::date_time::Timestamp& timestamp ) const
{
	String s = DoToLongDateTimeMillisString(timestamp);
	s.trim();
	assert(!s.empty() && "Invalid date formatting.");
	return s;
}

axis::foundation::date_time::Timestamp axis::services::locales::DateTimeFacet::ToLocalTime( const axis::foundation::date_time::Timestamp& timestamp ) const
{
	// ignore if timestamp is already in local time
	if (timestamp.IsLocalTime()) return timestamp;

	Timestamp localTime = DoToLocalTime(timestamp);
	Timespan diff = localTime - timestamp;
	assert(diff.GetDays() == 0 && "Unexpected timestamp computation behavior: wrong DST calculation!");
	assert(localTime.IsLocalTime() && "Unexpected timestamp computation behavior: local time attribute lost!");

	return localTime;
}
