// axis.PerformanceTestProject.cpp : Defines the entry point for the console application.
//
#include "stdafx.h"
#include <iostream>
#include "string_traits.hpp"
#include "application/runnable/AxisApplication.hpp"
#include "services/locales/LocaleLocator.hpp"
#include "services/logging/LogFile.hpp"
#include "services/io/FileSystem.hpp"
#include "application/jobs/StructuralAnalysis.hpp"
#include "domain/analyses/NumericalModel.hpp"
#include "domain/algorithms/Solver.hpp"
#include "application/output/collectors/filters/ApplicationEventFilter.hpp"
#include "application/output/collectors/filters/AnalysisEventFilter.hpp"
#include "application/jobs/JobRequest.hpp"
#include "System.hpp"
#include "foundation/memory/HeapBlockArena.hpp"
#include "foundation/memory/HeapStackArena.hpp"
#include <omp.h>
#include <direct.h>

namespace aaj = axis::application::jobs;
namespace aaocf = axis::application::output::collectors::filters;
namespace aar = axis::application::runnable;
namespace ad = axis::domain;
namespace asio = axis::services::io;
namespace asmm = axis::services::messaging;
namespace aslc = axis::services::locales;
namespace aslg = axis::services::logging;
namespace afb = axis::foundation::blas;
namespace afd = axis::foundation::date_time;

// const axis::String::char_type * ConfigFileLocation = _T("test_settings.config");
const axis::String::char_type * ConfigFileLocation = _T("axis.config");
// const axis::String::char_type * OutputFolderLocation = _T(".");
// const axis::String::char_type * OutputLogLocation = _T("06_steel_plasticity_I.log");
const axis::String::char_type * AppLogLocation = _T("test_app.log");
// const axis::String::char_type * baseSamplesDirLocation = _T("Model Input Files/nonlinear_explicit_dynamic/06_steel_plasticity_I"); 
// const axis::String::char_type * analysisFile = _T("06_input_file.axis");

void PrintMemoryStatistics(const axis::String& headerTitle)
{
  uint64 totalStringSpace = axis::System::StringMemory().GetTotalSize();
  uint64 totalModelSpace = axis::System::ModelMemory().GetTotalSize();
  uint64 totalGlobalSpace = axis::System::GlobalMemory().GetTotalSize();
  uint64 usedStringSpace = totalStringSpace - axis::System::StringMemory().GetNonContiguousAllocatedFreeSpace();
  uint64 usedModelSpace = totalModelSpace - axis::System::ModelMemory().GetNonContiguousAllocatedFreeSpace();
  uint64 usedGlobalSpace = totalGlobalSpace - axis::System::GlobalMemory().GetNonContiguousAllocatedFreeSpace();
  uint64 percStringSpace = usedStringSpace*10000 / totalStringSpace;
  uint64 percModelSpace = usedModelSpace*10000 / totalModelSpace;
  uint64 percGlobalSpace = usedGlobalSpace*10000 / totalGlobalSpace;

  std::wcout << std::endl;
  std::wcout << headerTitle.align_center(78) << std::endl;
  std::wcout << _T("=============================== MEMORY SUMMARY ===============================") << std::endl;
  std::wcout << _T("                         STRING MEMORY       MODEL MEMORY      GLOBAL MEMORY  ") << std::endl;
  std::wcout << _T("==============================================================================") << std::endl;
  std::wcout << _T(" Allocated size          ") << axis::String::int_parse((long)totalStringSpace / 1000).align_right(10) << _T(" KB")
    << _T("      ")          << axis::String::int_parse((long)totalModelSpace / 1000).align_right(10) << _T(" KB")
    << _T("      ")          << axis::String::int_parse((long)totalGlobalSpace / 1000).align_right(10) << _T(" KB") << std::endl;
  std::wcout << _T(" Fragmentation count     ") << axis::String::int_parse(axis::System::StringMemory().GetFragmentationCount()).align_right(13)
    << _T("      ")         << axis::String::int_parse(axis::System::ModelMemory().GetFragmentationCount()).align_right(13) 
    << _T("      ")         << axis::String::int_parse(axis::System::GlobalMemory().GetFragmentationCount()).align_right(13) << std::endl;
  std::wcout << _T(" Max. contg. free space  ") << axis::String::int_parse((long)axis::System::StringMemory().GetMaxContiguousAllocatedFreeSpace() / 1000).align_right(10) << _T(" KB")
    << _T("      ")         << axis::String::int_parse((long)axis::System::ModelMemory().GetMaxContiguousAllocatedFreeSpace() / 1000).align_right(10) << _T(" KB")
    << _T("      ")         << axis::String::int_parse((long)axis::System::GlobalMemory().GetMaxContiguousAllocatedFreeSpace() / 1000).align_right(10) << _T(" KB") << std::endl;
  std::wcout << _T(" Total free space        ") << axis::String::int_parse((long)axis::System::StringMemory().GetNonContiguousAllocatedFreeSpace() / 1000).align_right(10) << _T(" KB")
    << _T("      ")         << axis::String::int_parse((long)axis::System::ModelMemory().GetNonContiguousAllocatedFreeSpace() / 1000).align_right(10) << _T(" KB")
    << _T("      ")         << axis::String::int_parse((long)axis::System::GlobalMemory().GetNonContiguousAllocatedFreeSpace() / 1000).align_right(10) << _T(" KB") << std::endl;
  std::wcout << _T("------------------------------------------------------------------------------") << std::endl;
  std::wcout << _T(" Total used space        ") << axis::String::int_parse((long)usedStringSpace / 1000).align_right(10) << _T(" KB")
    << _T("      ")         << axis::String::int_parse((long)usedModelSpace / 1000).align_right(10) << _T(" KB")
    << _T("      ")         << axis::String::int_parse((long)usedGlobalSpace / 1000).align_right(10) << _T(" KB") << std::endl;
  std::wcout << _T(" Usage percentage        ") << axis::String::int_parse((long)percStringSpace/100).align_right(9) << _T(".") << axis::String::int_parse((long)percStringSpace%100).align_right(2).replace(_T(" "), _T("0")) << _T("%")
    << _T("      ")         << axis::String::int_parse((long)percModelSpace/100).align_right(9) << _T(".") << axis::String::int_parse((long)percModelSpace%100).align_right(2).replace(_T(" "), _T("0")) << _T("%")
    << _T("      ")         << axis::String::int_parse((long)percGlobalSpace/100).align_right(9) << _T(".") << axis::String::int_parse((long)percGlobalSpace%100).align_right(2).replace(_T(" "), _T("0")) << _T("%") << std::endl;
  std::wcout << _T("==============================================================================") << std::endl << std::endl;
}

int _tmain(int argc, _TCHAR* argv[])
{
  if (argc < 3)
  {
    std::wcout << _T("ERROR: too few arguments!") << std::endl;
    std::wcout << _T("USAGE: axis-mef <base folder> <input file>") << std::endl;
    exit(-1);
  }
  wchar_t curWorkingDirPath[10241];
  _wgetcwd(curWorkingDirPath, 10240);

  std::wcout << _T("==============================================================================") << std::endl;
  std::wcout << _T("      A X I S   F E A   F O R   S T R U C T U R A L   M E C H A N I C S       ") << std::endl;
  std::wcout << _T("==============================================================================") << std::endl;
#ifdef DEBUG
  std::wcout << _T("                                          -- CHANGESET 780 2014-04-03 18:37   ") << std::endl << std::endl;
#else
  std::wcout << _T("  ** RELEASE VERSION **                   -- CHANGESET 780 2014-04-03 18:37   ") << std::endl << std::endl;
#endif
  std::wcout << _T("DEVELOPED BY RENATO T. YAMASSAKI, 2014                                        ") << std::endl << std::endl << std::endl;

  std::wcout << _T("STATUS: Starting application...") << std::endl << std::endl;

  axis::System::Initialize();
  aar::AxisApplication& app = aar::AxisApplication::CreateApplication();
  const aslc::Locale& loc = aslc::LocaleLocator::GetLocator().GetGlobalLocale();
  axis::String curTimeString;
  axis::String curWorkingDirPathStr(curWorkingDirPath);
  afd::Timestamp currentTime;

  std::wcout << _T("  -- DIAGNOSTIC INFORMATION ----------------                                  ") << std::endl;
  std::wcout << _T("    OMP NUM THREADS = ") << omp_get_max_threads() << std::endl;
  std::wcout << _T("    AXIS_FORCE_CPU  = ") << axis::System::Environment().ForceCPU() << std::endl;
  std::wcout << _T("    AXIS_FORCE_GPU  = ") << axis::System::Environment().ForceGPU() << std::endl;
  std::wcout << _T("    BASE_DIR_PATH   = ") << curWorkingDirPathStr << std::endl;
  std::wcout << _T("    CONFIG_FILENAME = ") << ConfigFileLocation << std::endl;
  std::wcout << _T("  ------------------------------------------") << std::endl << std::endl;


  std::wcout << _T("STATUS: Opening log files for write operation...") << std::endl;

  // Build file input/output location
  axis::String workDir = asio::FileSystem::ToCanonicalPath(argv[1]);
  axis::String inputFile = asio::FileSystem::ConcatenatePath(workDir, argv[2]);
  axis::String fileTitle = asio::FileSystem::GetFileTitle(argv[2]);
  axis::String analysisLogLocation = asio::FileSystem::ConcatenatePath(workDir, fileTitle + _T(".log"));


  std::wcout << _T("  -- JOB PARAMETERS ----------------------------") << std::endl;
  std::wcout << _T("    Application log location      : ") << AppLogLocation << std::endl;
  std::wcout << _T("    Base search and output folder : ") << workDir << std::endl;
  std::wcout << _T("    Analysis log location         : ") << analysisLogLocation << std::endl;
  std::wcout << _T("    Analysis file                 : ") << inputFile << std::endl;
  std::wcout << _T("  ----------------------------------------------") << std::endl << std::endl;

  // start application and output log
  aslg::LogFile& appLog      = aslg::LogFile::Create(AppLogLocation);
  aslg::LogFile& analysisLog = aslg::LogFile::Create(analysisLogLocation);
  std::wcout << _T("STATUS: Log files created/opened successfully!") << std::endl;
  
  // apply proper filters to logs
  aaocf::ApplicationEventFilter appFilter;
  aaocf::AnalysisEventFilter analysisFilter;
  appLog.ReplaceFilter(appFilter);
  analysisFilter.SetMinInfoLevel(asmm::InfoMessage::InfoDebugLevel);
  analysisLog.ReplaceFilter(analysisFilter);
  std::wcout << _T("STATUS: Message filtering set! ") << std::endl;

  // connect logs to application output
  app.ConnectListener(appLog);
  app.ConnectListener(analysisLog);
  appLog.StartLogging();
  analysisLog.StartLogging();
  std::wcout << _T("STATUS: Logging started!") << std::endl;


  std::wcout << _T("======== BOOTSTRAP PHASE START =========================================") << std::endl;
  std::wcout << _T("STATUS: System bootstrap now! ") << std::endl;
  app.Configuration().SetConfigurationScriptPath(
                            asio::FileSystem::ConcatenatePath(
                            asio::FileSystem::GetApplicationFolder(), ConfigFileLocation));
  app.Bootstrap();
  std::wcout << _T("======== BOOTSTRAP PHASE END ===========================================") << std::endl << std::endl;


  if (!asio::FileSystem::ExistsFile(inputFile))
  {	// fail if file does not exist
    std::wcerr << _T("ERROR! File not found: ") << inputFile << std::endl;
    return 1;
  }

  std::wcout << _T("======== ANALYSIS PARSING PHASE START ==================================") << std::endl;
  std::wcout << _T("STATUS: Going to read input file now!") << std::endl;
  currentTime = afd::Timestamp::GetLocalTime();
  curTimeString = loc.GetDataTimeLocale().ToLongDateTimeMillisString(currentTime);
  std::wcout << _T("STATUS: Current local time is ") << curTimeString << std::endl;
  aaj::JobRequest job(inputFile, workDir, workDir);
  afd::Timestamp dt1 = afd::Timestamp::GetUTCTime();
  app.SubmitJob(job);
  afd::Timestamp dt2 = afd::Timestamp::GetUTCTime();
  afd::Timespan readTime = dt2 - dt1;
  std::wcout << _T("STATUS: Read in ") << readTime.GetTotalSeconds() << _T(".") << 
                axis::String::int_parse(readTime.GetMilliseconds(), 3).replace(_T(" "), _T("0")) << _T(" seconds.") << std::endl;
  PrintMemoryStatistics(_T("MEMORY STATISTICS AFTER PARSING"));
  currentTime = afd::Timestamp::GetLocalTime();
  curTimeString = loc.GetDataTimeLocale().ToLongDateTimeMillisString(currentTime);
  std::wcout << _T("STATUS: Current local time is ") << curTimeString << std::endl;
  std::wcout << _T("======== ANALYSIS PARSING PHASE END ====================================") << std::endl << std::endl;

  if (argc == 4)
  {
    if (argv[3] == _T("--dry-run"))
    {
      std::wcout << _T("STATUS: Dry run ended!") << curTimeString << std::endl;
      exit(0);
    }
  }

  // run analysis
  std::wcout << _T("======== ANALYSIS PROCESSING PHASE START ===============================") << std::endl;
  std::wcout << _T("STATUS: Starting analysis...") << std::endl;
  currentTime = afd::Timestamp::GetLocalTime();
  curTimeString = loc.GetDataTimeLocale().ToLongDateTimeMillisString(currentTime);
  std::wcout << _T("STATUS: Current local time is ") << curTimeString << std::endl;
  afd::Timestamp rt1 = afd::Timestamp::GetUTCTime();
  app.RunCurrentJob();
  afd::Timestamp rt2 = afd::Timestamp::GetUTCTime();
  afd::Timespan simulationTime = rt2 - rt1;
  std::wcout << _T("STATUS: Ran in ") << simulationTime.GetTotalSeconds() << _T(".") << 
             axis::String::int_parse(simulationTime.GetMilliseconds(), 3).replace(_T(" "), _T("0")) << _T(" seconds.") << std::endl;
  currentTime = afd::Timestamp::GetLocalTime();
  curTimeString = loc.GetDataTimeLocale().ToLongDateTimeMillisString(currentTime);
  std::wcout << _T("STATUS: Current local time is ") << curTimeString << std::endl;

  // stop logging
  appLog.StopLogging();
  analysisLog.StopLogging();

  PrintMemoryStatistics(_T("MEMORY STATISTICS AFTER ANALYSIS"));

  std::wcout << _T("STATUS: Analysis finished!") << std::endl;
  std::wcout << _T("======== ANALYSIS PROCESSING PHASE END =================================") << std::endl << std::endl;

  appLog.Destroy();
  analysisLog.Destroy();

//   std::cin.sync();
//   std::cin.ignore();
  return 0;
}

