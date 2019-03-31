#include "Process.hpp"
#include "ProcessPimpl.hpp"

axis::services::diagnostics::Process::Process( long processId )
{
	_data = new ProcessData((HANDLE)processId);
	Refresh();
}

axis::services::diagnostics::Process::Process( const Process& other )
{
	_data = new ProcessData(*other._data);
}

axis::services::diagnostics::Process::Process( void )
{
	_data = new ProcessData();
}

axis::services::diagnostics::Process::~Process( void )
{
	delete _data;
}

axis::services::diagnostics::Process& axis::services::diagnostics::Process::operator=( const Process& other )
{
	ProcessData *newData = new ProcessData(*other._data);
	delete _data;
	_data = newData;

	return *this;
}

axis::services::diagnostics::Process axis::services::diagnostics::Process::GetCurrentProcess( void )
{
	return Process((long)::GetCurrentProcess());
}

void axis::services::diagnostics::Process::Refresh( void )
{
	_data->Refresh();
}

long axis::services::diagnostics::Process::GetProcessId( void ) const
{
	return (long)_data->GetProcessId();
}

axis::foundation::date_time::Timespan axis::services::diagnostics::Process::GetKernelTime( void ) const
{
	return _data->GetKernelTime();
}

axis::foundation::date_time::Timespan axis::services::diagnostics::Process::GetUserTime( void ) const
{
	return _data->GetUserTime();
}

axis::foundation::date_time::Timespan axis::services::diagnostics::Process::GetWallTime( void ) const
{
	Timestamp now = Timestamp::GetLocalTime();
	Timespan wallTime = now - _data->GetCreationDate();
	return wallTime;
}

axis::foundation::date_time::Timestamp axis::services::diagnostics::Process::GetCreationTime( void ) const
{
	return _data->GetCreationDate();
}

uint64 axis::services::diagnostics::Process::GetPhysicalMemoryAllocated( void ) const
{
	return _data->GetPhysicalMemoryUsageInBytes();
}

uint64 axis::services::diagnostics::Process::GetVirtualMemoryAllocated( void ) const
{
	return _data->GetVirtualMemoryUsageInBytes();
}

uint64 axis::services::diagnostics::Process::GetPagedMemoryAllocated( void ) const
{
	return _data->GetPagedMemoryUsageInBytes();
}

uint64 axis::services::diagnostics::Process::GetPeakPhysicalMemoryAllocated( void ) const
{
	return _data->GetPeakPhysicalMemoryUsageInBytes();
}

uint64 axis::services::diagnostics::Process::GetPeakVirtualMemoryAllocated( void ) const
{
	return _data->GetPeakVirtualMemoryUsageInBytes();
}

uint64 axis::services::diagnostics::Process::GetPeakPagedMemoryAllocated( void ) const
{
	return _data->GetPeakPagedMemoryUsageInBytes();
}

double axis::services::diagnostics::Process::GetCPUUsage( void ) const
{
	return _data->GetCPUUsagePercent();
}

double axis::services::diagnostics::Process::GetCPUOccupation( void ) const
{
	return _data->GetCPUOccupationPercent();
}

double axis::services::diagnostics::Process::GetUserCodeOccupation( void ) const
{
	return _data->GetUserCodeOccupationPercent();
}

double axis::services::diagnostics::Process::GetTotalCPUUsage( void ) const
{
	return _data->GetTotalCPUUsagePercent();
}

double axis::services::diagnostics::Process::GetTotalCPUOccupation( void ) const
{
	return _data->GetTotalCPUOccupationPercent();
}

double axis::services::diagnostics::Process::GetTotalUserCodeOccupation( void ) const
{
	return _data->GetTotalUserCodeOccupationPercent();
}
