#include "Date.hpp"
#include <boost/date_time/gregorian/gregorian.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include "foundation/OutOfRangeException.hpp"
#include "foundation/ArgumentException.hpp"

using namespace boost::gregorian;
using namespace boost::posix_time;

// Pimpl idiom
class axis::foundation::date_time::Date::DateData
{
private:
	boost::gregorian::date _date;
public:
	DateData(void) : _date(date::year_type(1900), date::month_type(1), date::day_type(1))
	{
		// nothing to do here
	}

	DateData(const boost::gregorian::date& date) : _date(date)
	{
		// nothing to do here
	}

	void Set(const boost::gregorian::date& newDate)
	{
		_date = newDate;
	}

	boost::gregorian::date GetDate(void) const
	{
		return _date;
	}

	int GetDay(void) const
	{
		return (int)_date.day();
	}

	int GetMonth(void) const
	{
		return (int)_date.month();
	}

	int GetYear(void) const
	{
		return (int)_date.year();
	}

	int GetDayOfWeek(void) const
	{
		return (int)_date.day_of_week();
	}
	
	int GetDayOfYear(void) const
	{
		return (int)_date.day_of_year();
	}

	int GetWeekOfYear(void) const
	{
		return (int)_date.week_number();
	}
};



axis::foundation::date_time::Date::Date( void )
{
	_data = new DateData();
}

axis::foundation::date_time::Date::Date( const axis::foundation::date_time::Date& other )
{
	_data = new DateData();
	_data->Set(other._data->GetDate());
}

axis::foundation::date_time::Date::Date( int year, int month, int day )
{
	_data = new DateData();
	try
	{
		_data->Set(date(date::year_type(year), date::month_type(month), date::day_type(day)));
	}
	catch (std::out_of_range&)
	{
		// invalid date
		throw axis::foundation::OutOfRangeException(_T("Invalid date."));
	}
}

axis::foundation::date_time::Date::~Date( void )
{
	delete _data;
}

int axis::foundation::date_time::Date::GetDay( void ) const
{
	return _data->GetDay();
}

int axis::foundation::date_time::Date::GetMonth( void ) const
{
	return _data->GetMonth();
}

int axis::foundation::date_time::Date::GetYear( void ) const
{
	return _data->GetYear();
}

axis::foundation::date_time::Date::WeekDay axis::foundation::date_time::Date::GetDayOfWeek( void ) const
{
	return (WeekDay)_data->GetDayOfWeek();
}

int axis::foundation::date_time::Date::GetDayOfYear( void ) const
{
	return _data->GetDayOfYear();
}

int axis::foundation::date_time::Date::GetWeekOfYear( void ) const
{
	return _data->GetWeekOfYear();
}

axis::foundation::date_time::Date& axis::foundation::date_time::Date::operator=( const axis::foundation::date_time::Date& other )
{
	if (this == &other) return *this;
	_data->Set(other._data->GetDate());
	return *this;
}

bool axis::foundation::date_time::Date::operator<( const axis::foundation::date_time::Date& other ) const
{
	return _data->GetDate() < other._data->GetDate();
}

bool axis::foundation::date_time::Date::operator<=( const axis::foundation::date_time::Date& other ) const
{
	return _data->GetDate() <= other._data->GetDate();
}

bool axis::foundation::date_time::Date::operator>( const axis::foundation::date_time::Date& other ) const
{
	return _data->GetDate() > other._data->GetDate();
}

bool axis::foundation::date_time::Date::operator>=( const axis::foundation::date_time::Date& other ) const
{
	return _data->GetDate() >= other._data->GetDate();
}

bool axis::foundation::date_time::Date::operator==( const axis::foundation::date_time::Date& other ) const
{
	return _data->GetDate() == other._data->GetDate();
}

bool axis::foundation::date_time::Date::operator!=( const axis::foundation::date_time::Date& other ) const
{
	return _data->GetDate() != other._data->GetDate();
}

axis::foundation::date_time::Date axis::foundation::date_time::Date::Today( void )
{
	ptime p = second_clock::local_time();
	date dt = p.date();
	return Date(dt.year(), dt.month(), dt.day());
}

axis::foundation::date_time::Date axis::foundation::date_time::operator+( const Date& date, const Timespan& dateRange )
{
	if (dateRange.HasFractionalDay())
	{
		throw axis::foundation::ArgumentException(_T("Cannot operate on fractional days."));
	}

	boost::gregorian::date dt(date::year_type(date.GetYear()), date::month_type(date.GetMonth()), date::day_type(date.GetDay()));	
	dt += date_duration(dateRange.GetDays());
	return Date(dt.year(), dt.month(), dt.day());
}

axis::foundation::date_time::Date axis::foundation::date_time::operator-( const Date& date, const Timespan& dateRange )
{
	if (dateRange.HasFractionalDay())
	{
		throw axis::foundation::ArgumentException(_T("Cannot operate on fractional days."));
	}

	boost::gregorian::date dt(date::year_type(date.GetYear()), date::month_type(date.GetMonth()), date::day_type(date.GetDay()));	
	dt -= date_duration(dateRange.GetDays());
	if (dt.is_not_a_date())
	{
		throw axis::foundation::OutOfRangeException(_T("Result is not a valid or supported date."));
	}	
	return Date(dt.year(), dt.month(), dt.day());
}

axis::foundation::date_time::Timespan axis::foundation::date_time::operator-( const Date& d1, const Date& d2 )
{
	date dt1(date::year_type(d1.GetYear()), date::month_type(d1.GetMonth()), date::day_type(d1.GetDay()));	
	date dt2(date::year_type(d2.GetYear()), date::month_type(d2.GetMonth()), date::day_type(d2.GetDay()));	
	date_duration d = dt1 - dt2;
	return Timespan(d.days(), Timespan::Days);
}
