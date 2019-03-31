#include "ApplicationEnvironment.hpp"
#include <windows.h>
#include <omp.h>
#include "foundation/OutOfMemoryException.hpp"
#include "foundation/ApplicationErrorException.hpp"
#include <string>
#include <boost/algorithm/string.hpp> 


#define AXIS_FORCE_GPU    "AXIS_FORCE_GPU"
#define AXIS_FORCE_CPU    "AXIS_FORCE_CPU"

class axis::ApplicationEnvironment::Pimpl
{
public:
  bool ForceGPU;
  bool ForceCPU;

  Pimpl(void)
  {
    ForceCPU = false;
    ForceCPU = false;
  }
};

namespace {
  typedef struct 
  {
    long PhysicalProcessorCount;
    long LogicalProcessorCount;
    long ProcessorCoreCount;
  } ProcessorStatistics;

  void ResolveBooleanValue(bool& targetVariable, const char *value, bool defaultValue = false)
  {
    if (value == nullptr) 
    {
      targetVariable = defaultValue;
      return;
    }
    std::string s = value;
    boost::trim(s);
    boost::to_lower(s);
	if (s.length() > 1)
	{
		if (s[0] == '\"' && s[s.length() - 1] == '\"')
		{
			s = s.substr(1, s.length() - 2);
		}
	}

    if (s == "yes" || s == "true" || s == "1")
    {
      targetVariable = true;
    }
    else if (s == "no" || s == "false" || s == "0")
    {
      targetVariable = false;
    }
    else
    { // undefined value
      targetVariable = defaultValue;
    }
  }


  // Helper function to count set bits in the processor mask.
  // Extracted from MSDN GetLogicalProcessorInformation article.
  DWORD CountSetBits(ULONG_PTR bitMask)
  {
    DWORD LSHIFT = sizeof(ULONG_PTR)*8 - 1;
    DWORD bitSetCount = 0;
    ULONG_PTR bitTest = (ULONG_PTR)1 << LSHIFT;    
    DWORD i;

    for (i = 0; i <= LSHIFT; ++i)
    {
      bitSetCount += ((bitMask & bitTest)?1:0);
      bitTest/=2;
    }

    return bitSetCount;
  }

  // Helper function that gets logical processors information
  // from Windows API
  DWORD GetProcessorsInformation(PSYSTEM_LOGICAL_PROCESSOR_INFORMATION *buffer)
  {
    *buffer = NULL;
    DWORD returnLength = 0;
    long cpuCount = 0;
    bool done = false;

    while (!done)
    {
      DWORD rc = GetLogicalProcessorInformation(*buffer, &returnLength);

      if (FALSE == rc) 
      {	// at first try, should fail and return required buffer size
        if (GetLastError() == ERROR_INSUFFICIENT_BUFFER) 
        {	// what we were expecting; allocate buffer
          if (*buffer) free(*buffer);
          *buffer = (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION)malloc(returnLength);
          if (NULL == buffer) 
          {	// could not allocate enough memory
            throw axis::foundation::OutOfMemoryException();
          }
        } 
        else 
        {	// huh, another error was triggered; this is bad...
          throw axis::foundation::ApplicationErrorException(
            _T("Could not retrieve local logical processor information. Error code = 0x") + 
            axis::String::int_to_hex(rc));
        }
      } 
      else
      {	// ok, processor information retrieved
        done = true;
      }	
    }

    return returnLength;
  }

  // Compiles processor information into a single statistics structure.
  ProcessorStatistics GetProcessorStatistics(void)
  {
    ProcessorStatistics statistics = {0};
    PSYSTEM_LOGICAL_PROCESSOR_INFORMATION buffer = NULL;
    DWORD returnLength = 0;
    int cpuCount = 0;

    // get information using helper function
    returnLength = GetProcessorsInformation(&buffer);

    // count how many processor records we have
    int recordCount = returnLength / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION);
    PSYSTEM_LOGICAL_PROCESSOR_INFORMATION ptr = buffer;
    for (int i = 0; i < recordCount; ++i)
    {
      // note: if in the future new relationships values are added,
      // this method might not count processors correctly
      if (ptr->Relationship == RelationProcessorCore)
      {
        // hyperthread cores supply more than one logical processor per
        // core
        statistics.LogicalProcessorCount += CountSetBits(ptr->ProcessorMask);

        // count a physical core
        statistics.ProcessorCoreCount++;
      }
      if (ptr->Relationship == RelationProcessorPackage)
      {
        // count a processor package
        statistics.PhysicalProcessorCount++;
      }

      // advance to next record
      ++ptr;
    }

    // free resources
    free(buffer);

    return statistics;
  }
}

axis::ApplicationEnvironment::ApplicationEnvironment( void )
{
  pimpl_ = new Pimpl();
}

axis::ApplicationEnvironment::~ApplicationEnvironment( void )
{
  delete pimpl_;
}

bool axis::ApplicationEnvironment::ForceGPU( void ) const
{
  return pimpl_->ForceGPU;
}

bool axis::ApplicationEnvironment::ForceCPU( void ) const
{
  return pimpl_->ForceCPU;
}

void axis::ApplicationEnvironment::RefreshFromSystem( void )
{
  const char *val = getenv(AXIS_FORCE_CPU);
  ResolveBooleanValue(pimpl_->ForceCPU, val, false);
  val = getenv(AXIS_FORCE_GPU);
  ResolveBooleanValue(pimpl_->ForceGPU, val, false);
}

int axis::ApplicationEnvironment::GetLocalLogicalProcessorCount( void )
{	
  return GetProcessorStatistics().LogicalProcessorCount;
}

int axis::ApplicationEnvironment::GetLocalProcessorCoreCount( void )
{
  return GetProcessorStatistics().ProcessorCoreCount;
}

int axis::ApplicationEnvironment::GetLocalProcessorPackageCount( void )
{
  return GetProcessorStatistics().PhysicalProcessorCount;
}

int axis::ApplicationEnvironment::GetLocalLogicalProcessorAvailableCount( void )
{
  // OpenMP is the one who restricts processors available for use
  return omp_get_num_procs();
}

int axis::ApplicationEnvironment::GetMaxLocalWorkerThreads( void )
{
  return omp_get_max_threads();
}

void axis::ApplicationEnvironment::SetMaxLocalWorkerThread( int threadCount )
{
  omp_set_num_threads(threadCount);
}

