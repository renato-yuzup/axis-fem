#pragma once
#include "foundation/Axis.SystemBase.hpp"

namespace axis { 

class AXISSYSTEMBASE_API ApplicationEnvironment
{
public:
  ~ApplicationEnvironment(void);

  bool ForceGPU(void) const;
  bool ForceCPU(void) const;

  
	/**
		* Returns the number of local logical processors, that is, processing units as interpreted by
		* the operating system.
		*
		* @return The local logical processor count.
		*/
	static int GetLocalLogicalProcessorCount(void);

	/**
		* Returns the total number of physical processors cores in all local processors.
		*
		* @return The local processor core count.
		*/
	static int GetLocalProcessorCoreCount(void);

	/**
		* Returns the number of local processor packages installed.
		*
		* @return The local processor package count.
		*/
	static int GetLocalProcessorPackageCount(void);				

	/**
		* Returns the number of local logical processors which are available for use by this
		* application instance.
		*
		* @return The local logical processor available count.
		*/
	static int GetLocalLogicalProcessorAvailableCount(void);

	/**
		* Returns the maximum number of simultaneous worker threads this application instance can own.
		*
		* @return The maximum local worker threads.
		*/
	static int GetMaxLocalWorkerThreads(void);

	/**
		* Sets the maximum number of simultaneous worker threads this application instance can own.
		*
		* @param threadCount The maximum number of threads allowed.
		*/
	static void SetMaxLocalWorkerThread(int threadCount);

  void RefreshFromSystem(void);
private:
  class Pimpl;
  Pimpl *pimpl_;
  ApplicationEnvironment(void);
  friend class System;
};

} // namespace axis
