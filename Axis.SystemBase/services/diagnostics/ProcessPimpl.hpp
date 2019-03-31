#pragma once

/*
	Compile with PSAPI_VERSION = 1 to maintain backward 
	compatibility with previous versions of Windows.
*/
#define PSAPI_VERSION			1
#include <windows.h>
#include <Psapi.h>

#include "Process.hpp"

#include "foundation/ApplicationErrorException.hpp"
#include "foundation/InvalidOperationException.hpp"

#include "foundation/date_time/Timespan.hpp"
#include "foundation/date_time/Timestamp.hpp"



using namespace axis::foundation::date_time;

namespace {
	inline uint64 FileTimeToUInt64(const FILETIME& t)
	{
		return (((uint64)t.dwHighDateTime) << 32) + (uint64)t.dwLowDateTime;
	}


	uint64 SubtractTimes(const FILETIME& ta, const FILETIME& tb)
	{
		uint64 a = FileTimeToUInt64(ta);
		uint64 b = FileTimeToUInt64(tb);
		return a - b;
	}
}

class axis::services::diagnostics::Process::ProcessData
{
private:
	// General process information
	HANDLE    _processHandle;
	long      _processId;
	Timestamp _procCreationTime;	

	// Process and system timing information
	uint64    _procKernelTime;
	uint64    _procUserTime;
	uint64    _sysIdleTime;
	uint64    _sysKernelTime;
	uint64    _sysUserTime;

	// Last run process and system timings
	uint64    _lastProcKernelTime;
	uint64    _lastProcUserTime;
	uint64    _lastSysIdleTime;
	uint64    _lastSysKernelTime;
	uint64    _lastSysUserTime;

	// System timings on the first information refresh
	uint64    _baseSysIdleTime;
	uint64    _baseSysKernelTime;
	uint64    _baseSysUserTime;

	// Process activity measurements
	double    _cpuUsagePercent;
	double    _cpuOccupationPercent;
	double    _userCodeOccupationPercent;
	double    _cpuTotalUsagePercent;
	double    _cpuTotalOccupationPercent;
	double    _totalUserCodeOccupationPercent;

	// Process memory usage
	uint64    _physicalMemoryUsage;
	uint64    _virtualMemoryUsage;
	uint64    _pagedMemoryUsage;
	uint64    _peakPhysicalMemoryUsage;
	uint64    _peakVirtualMemoryUsage;
	uint64    _peakPagedMemoryUsage;

	bool _isValid;

	void UpdateProcessTimings(void);
	void UpdateSystemTimings(void);
	void UpdateProcessActivityMeasurement(void);
	void UpdateProcessMemoryUsage(void);
public:
	explicit ProcessData(HANDLE hProcess);
	ProcessData(void);

	void GetGeneralProcessInformation(void);

	void Refresh(void);

	long GetProcessId(void) const;
	Timestamp GetCreationDate(void) const;

	Timespan GetKernelTime(void) const;
	Timespan GetUserTime(void) const;

	double GetCPUUsagePercent(void) const;
	double GetCPUOccupationPercent(void) const;
	double GetUserCodeOccupationPercent(void) const;
	double GetTotalCPUUsagePercent(void) const;
	double GetTotalCPUOccupationPercent(void) const;
	double GetTotalUserCodeOccupationPercent(void) const;

	uint64 GetPhysicalMemoryUsageInBytes(void) const;
	uint64 GetVirtualMemoryUsageInBytes(void) const;
	uint64 GetPagedMemoryUsageInBytes(void) const;
	uint64 GetPeakPhysicalMemoryUsageInBytes(void) const;
	uint64 GetPeakVirtualMemoryUsageInBytes(void) const;
	uint64 GetPeakPagedMemoryUsageInBytes(void) const;

};

axis::services::diagnostics::Process::ProcessData::ProcessData( HANDLE hProcess ) : _processHandle(hProcess)
{
	_sysUserTime    = 0;
	_sysKernelTime  = 0;
	_sysIdleTime    = 0;
	_procUserTime   = 0;
	_procKernelTime = 0;

	_lastSysUserTime    = 0;
	_lastSysKernelTime  = 0;
	_lastSysIdleTime    = 0;
	_lastProcUserTime   = 0;
	_lastProcKernelTime = 0;

	_isValid = true;

	// Retrieve general process information
	GetGeneralProcessInformation();
	UpdateSystemTimings();
}

axis::services::diagnostics::Process::ProcessData::ProcessData( void )
{
	_isValid = false;
}

void axis::services::diagnostics::Process::ProcessData::GetGeneralProcessInformation( void )
{
	if (!_isValid)
	{
		throw axis::foundation::InvalidOperationException();
	}

	FILETIME ftCreationTime,	// creation time of the process
			 ftExitTime,		// <not used>
			 ftProcKernelTime,	// <not used>
			 ftProcUserTime;	// <not used>

	_processId = (long)::GetProcessId(_processHandle);

	BOOL ok = GetProcessTimes(_processHandle, &ftCreationTime, &ftExitTime, &ftProcKernelTime, &ftProcUserTime);
	if (ok == FALSE)
	{
		DWORD errId = GetLastError();
		throw axis::foundation::ApplicationErrorException(_T("Could not retrieve process timing information. Error code = 0x") + axis::String::int_to_hex(errId));
	}

	// convert FILETIME structures into system time (UTC) and then
	// into local time
	SYSTEMTIME stCreationTime, ltCreationTime;
	ok = FileTimeToSystemTime(&ftCreationTime, &stCreationTime);
	ok &= SystemTimeToTzSpecificLocalTime(NULL, &stCreationTime, &ltCreationTime);
	if (ok == FALSE)
	{
		throw axis::foundation::ApplicationErrorException(_T("Error on handling process timing structures."));
	}

	_procCreationTime = Timestamp(Date(ltCreationTime.wYear, ltCreationTime.wMonth, ltCreationTime.wDay),
								  Time(ltCreationTime.wHour, ltCreationTime.wMinute, ltCreationTime.wSecond, ltCreationTime.wMilliseconds));
}

void axis::services::diagnostics::Process::ProcessData::UpdateProcessTimings( void )
{
	if (!_isValid)
	{
		throw axis::foundation::InvalidOperationException();
	}

	FILETIME ftCreationTime,	// creation time of the process
			 ftExitTime,		// exit time
			 ftProcKernelTime,	// kernel time (sum of all thread times)
			 ftProcUserTime;	// user time (sum of all thread times)

	// retrieve process timings
	BOOL ok = GetProcessTimes(_processHandle, &ftCreationTime, &ftExitTime, &ftProcKernelTime, &ftProcUserTime);
	if (ok == FALSE)
	{
		DWORD errId = GetLastError();
		throw axis::foundation::ApplicationErrorException(_T("Could not retrieve process timing information. Error code = 0x") + axis::String::int_to_hex(errId));
	}

	// convert into a more flexible type
	uint64 curProcUserTime   = FileTimeToUInt64(ftProcUserTime);
	uint64 curProcKernelTime = FileTimeToUInt64(ftProcKernelTime);

	// update timings
	_lastProcKernelTime = _procKernelTime; 
	_lastProcUserTime   = _procUserTime; 

	_procKernelTime = curProcKernelTime;
	_procUserTime   = curProcUserTime;
}

void axis::services::diagnostics::Process::ProcessData::UpdateSystemTimings( void )
{
	if (!_isValid)
	{
		throw axis::foundation::InvalidOperationException();
	}

	FILETIME ftSysIdleTime,		// total system idle time (all processes)
			 ftSysKernelTime,	// total system kernel time (all processes)
			 ftSysUserTime;		// total system user time (all processes)

	// retrieve system timings
	BOOL ok = GetSystemTimes(&ftSysIdleTime, &ftSysKernelTime, &ftSysUserTime);
	if (ok == FALSE)
	{
		DWORD errId = GetLastError();
		throw axis::foundation::ApplicationErrorException(_T("Could not retrieve system timing information. Error code = 0x") + axis::String::int_to_hex(errId));
	}

	// convert into a more flexible type
	uint64 curSysIdleTime    = FileTimeToUInt64(ftSysIdleTime);
	uint64 curSysKernelTime  = FileTimeToUInt64(ftSysKernelTime);
	uint64 curSysUserTime    = FileTimeToUInt64(ftSysUserTime);

	// update timings
	if (_baseSysKernelTime == 0) // virtually impossible, so it means this is the first measurement
	{
		_baseSysKernelTime = curSysKernelTime;
		_baseSysUserTime = curSysUserTime;
		_baseSysIdleTime = curSysIdleTime;
	}

	_lastSysIdleTime    = _sysIdleTime; 
	_lastSysKernelTime  = _sysKernelTime; 
	_lastSysUserTime    = _sysUserTime; 

	_sysKernelTime  = curSysKernelTime;
	_sysUserTime    = curSysUserTime;
	_sysIdleTime    = curSysIdleTime;
}

void axis::services::diagnostics::Process::ProcessData::UpdateProcessActivityMeasurement( void )
{
	if (!_isValid)
	{
		throw axis::foundation::InvalidOperationException();
	}

	uint64 procUserTimeDiff   = _procUserTime - _lastProcUserTime;
	uint64 procKernelTimeDiff = _procKernelTime - _lastProcKernelTime;

	uint64 sysUserTimeDiff    = _sysUserTime - _lastSysUserTime;
	uint64 sysKernelTimeDiff  = _sysKernelTime - _lastSysKernelTime;
	uint64 sysIdleTimeDiff    = _sysIdleTime - _lastSysIdleTime;

	uint64 totalSysUserTime    = _sysUserTime - _baseSysUserTime;
	uint64 totalSysKernelTime  = _sysKernelTime - _baseSysKernelTime;
	uint64 totalSysIdleTime    = _sysIdleTime - _baseSysIdleTime;

	_cpuUsagePercent           = (double)(procUserTimeDiff + procKernelTimeDiff) / (double)(sysUserTimeDiff + sysKernelTimeDiff);
	_cpuOccupationPercent      = (double)(procUserTimeDiff + procKernelTimeDiff) / (double)(sysUserTimeDiff + sysKernelTimeDiff + sysIdleTimeDiff);
	_userCodeOccupationPercent = (double)procUserTimeDiff / (double)(procUserTimeDiff + procKernelTimeDiff);

	_cpuTotalUsagePercent = (double)(_procUserTime + _procKernelTime) / (double)(totalSysUserTime + totalSysKernelTime);
	_cpuTotalOccupationPercent = (double)(_procUserTime + _procKernelTime) / (double)(totalSysUserTime + totalSysKernelTime + totalSysIdleTime);
	_totalUserCodeOccupationPercent = (double)_procUserTime / (double)(_lastProcUserTime + _lastProcKernelTime);
}

void axis::services::diagnostics::Process::ProcessData::UpdateProcessMemoryUsage( void )
{
	if (!_isValid)
	{
		throw axis::foundation::InvalidOperationException();
	}

	PROCESS_MEMORY_COUNTERS mem_info = {0};
	mem_info.cb = sizeof(PROCESS_MEMORY_COUNTERS);

	if (!GetProcessMemoryInfo(_processHandle, (PPROCESS_MEMORY_COUNTERS)&mem_info, mem_info.cb))
	{
		throw axis::foundation::ApplicationErrorException(_T("System call failed."));
	}

	_physicalMemoryUsage = mem_info.WorkingSetSize;
	_peakPhysicalMemoryUsage = mem_info.PeakWorkingSetSize;

	_virtualMemoryUsage = mem_info.PagefileUsage;
	_peakVirtualMemoryUsage = mem_info.PeakPagefileUsage;

	_pagedMemoryUsage = mem_info.QuotaPagedPoolUsage;
	_peakPagedMemoryUsage = mem_info.QuotaPeakPagedPoolUsage;
}

long axis::services::diagnostics::Process::ProcessData::GetProcessId( void ) const
{
	return _processId;
}

axis::foundation::date_time::Timestamp axis::services::diagnostics::Process::ProcessData::GetCreationDate( void ) const
{
	return _procCreationTime;
}

axis::foundation::date_time::Timespan axis::services::diagnostics::Process::ProcessData::GetKernelTime( void ) const
{
	return Timespan(_procKernelTime);
}

axis::foundation::date_time::Timespan axis::services::diagnostics::Process::ProcessData::GetUserTime( void ) const
{
	return Timespan(_procUserTime);
}

double axis::services::diagnostics::Process::ProcessData::GetCPUUsagePercent( void ) const
{
	return _cpuUsagePercent;
}

double axis::services::diagnostics::Process::ProcessData::GetCPUOccupationPercent( void ) const
{
	return _cpuOccupationPercent;
}

double axis::services::diagnostics::Process::ProcessData::GetUserCodeOccupationPercent( void ) const
{
	return _userCodeOccupationPercent;
}

double axis::services::diagnostics::Process::ProcessData::GetTotalCPUUsagePercent( void ) const
{
	return _cpuTotalUsagePercent;
}

double axis::services::diagnostics::Process::ProcessData::GetTotalCPUOccupationPercent( void ) const
{
	return _cpuTotalOccupationPercent;
}

double axis::services::diagnostics::Process::ProcessData::GetTotalUserCodeOccupationPercent( void ) const
{
	return _totalUserCodeOccupationPercent;
}

uint64 axis::services::diagnostics::Process::ProcessData::GetPhysicalMemoryUsageInBytes( void ) const
{
	return _physicalMemoryUsage;
}

uint64 axis::services::diagnostics::Process::ProcessData::GetVirtualMemoryUsageInBytes( void ) const
{
	return _virtualMemoryUsage;
}

uint64 axis::services::diagnostics::Process::ProcessData::GetPagedMemoryUsageInBytes( void ) const
{
	return _pagedMemoryUsage;
}

uint64 axis::services::diagnostics::Process::ProcessData::GetPeakPhysicalMemoryUsageInBytes( void ) const
{
	return _peakPhysicalMemoryUsage;
}

uint64 axis::services::diagnostics::Process::ProcessData::GetPeakVirtualMemoryUsageInBytes( void ) const
{
	return _peakVirtualMemoryUsage;
}

uint64 axis::services::diagnostics::Process::ProcessData::GetPeakPagedMemoryUsageInBytes( void ) const
{
	return _peakPagedMemoryUsage;
}

void axis::services::diagnostics::Process::ProcessData::Refresh( void )
{
	if (!_isValid)
	{
		throw axis::foundation::InvalidOperationException();
	}

	UpdateProcessTimings();
	UpdateSystemTimings();
	UpdateProcessActivityMeasurement();
	UpdateProcessMemoryUsage();
}

