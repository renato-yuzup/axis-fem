#include "ResultCollectorParser.hpp"
#include <assert.h>

#include "application/locators/CollectorFactoryLocator.hpp"
#include "application/locators/WorkbookFactoryLocator.hpp"
#include "application/output/ResultBucketConcrete.hpp"
#include "application/output/ResultDatabase.hpp"
#include "application/output/collectors/ElementSetCollector.hpp"
#include "application/output/collectors/EntityCollector.hpp"
#include "application/output/collectors/GenericCollector.hpp"
#include "application/output/collectors/NodeSetCollector.hpp"
#include "application/output/workbooks/ResultWorkbook.hpp"
#include "application/parsing/error_messages.hpp"
#include "application/parsing/warning_messages.hpp"
#include "application/parsing/core/SymbolTable.hpp"
#include "services/io/FileSystem.hpp"
#include "services/language/syntax/evaluation/ParameterList.hpp"
#include "services/messaging/ErrorMessage.hpp"
#include "services/messaging/InfoMessage.hpp"
#include "foundation/InvalidOperationException.hpp"
#include "foundation/NotSupportedException.hpp"

namespace aal = axis::application::locators;
namespace aapc = axis::application::parsing::core;
namespace aafc = axis::application::factories::collectors;
namespace aao = axis::application::output;
namespace aaoc = axis::application::output::collectors;
namespace aaow = axis::application::output::workbooks;
namespace aapps = axis::application::parsing::parsers;
namespace aapc = axis::application::parsing::core;
namespace aaj = axis::application::jobs;
namespace ada = axis::domain::analyses;
namespace asi = axis::services::io;
namespace asli = axis::services::language::iterators;
namespace aslp = axis::services::language::parsing;
namespace aslse = axis::services::language::syntax::evaluation;
namespace asmm = axis::services::messaging;

aapps::ResultCollectorParser::ResultCollectorParser( aal::WorkbookFactoryLocator& workbookLocator,
                                                     aal::CollectorFactoryLocator& locator, 
                                                     aao::ResultBucketConcrete& resultBucket,
                                                     const axis::String& fileName, 
                                                     const axis::String& formatName, 
                                                     const aslse::ParameterList& formatArguments,
                                                     bool append ) :
workbookLocator_(workbookLocator), collectorLocator_(locator), resultBucket_(resultBucket), 
formatArguments_(formatArguments.Clone()), fileName_(fileName), formatName_(formatName)
{
	onErrorRecovering_ = false;
	onSkipBuild_ = false;
	appendToFile_ = append;
  currentDatabase_ = NULL;
}

aapps::ResultCollectorParser::~ResultCollectorParser( void )
{
	formatArguments_.Destroy();
}

aapps::BlockParser& aapps::ResultCollectorParser::GetNestedContext( const axis::String&, 
                                                                    const aslse::ParameterList& )
{
	// it is not allowed to have nested blocks in the collector block
	throw axis::foundation::NotSupportedException();
}

void aapps::ResultCollectorParser::DoStartContext( void )
{
  // check if we know how to build the specified file format
  onErrorRecovering_ = !workbookLocator_.CanBuild(formatName_, formatArguments_);
  if (onErrorRecovering_)
  {
    axis::String s = AXIS_ERROR_MSG_UNRECOGNIZED_OUTPUT_FORMAT;
    s = s.replace(_T("%1"), formatName_);
    GetParseContext().RegisterEvent(asmm::ErrorMessage(AXIS_ERROR_ID_UNRECOGNIZED_OUTPUT_FORMAT, s));
  }
  else
  { // collectors will be added to this database
    currentDatabase_ = new axis::application::output::ResultDatabase();
  }
}

aslp::ParseResult aapps::ResultCollectorParser::Parse( const asli::InputIterator& begin, 
                                                       const asli::InputIterator& end )
{
	// check if we can parse so far
	aslp::ParseResult result = collectorLocator_.TryParse(formatName_, begin, end);

  if (result.GetResult() == aslp::ParseResult::FailedMatch)
  {	// failed to parse -- it rendered this block unusable
    onErrorRecovering_ = true;

    // trigger error
    axis::String s = AXIS_ERROR_MSG_INVALID_DECLARATION;
    GetParseContext().RegisterEvent(asmm::ErrorMessage(0x300502, s));
  }
	else if (result.IsMatch() && !onSkipBuild_)
	{	// yes, we can -- build collector (if no skip build flag was set)
		ada::NumericalModel& model = GetAnalysis().GetNumericalModel();
    aafc::CollectorBuildResult br = collectorLocator_.ParseAndBuild(model, GetParseContext(), 
                                                                    formatName_, begin, end);
		if (onErrorRecovering_)
		{	// ignore created collector
			if (br.Collector != NULL) br.Collector->Destroy();
		}
		else
		{ // if we are not on error recovery mode, add collector to list;
			// check if the collector was built; if not, this means that something
			// from the model is missing and we should skip building the collector chain
			if (br.IsModelIncomplete)
			{
				onSkipBuild_ = true;
			}
      else
			{ // add collector to the correct database chain
        switch (br.CollectorType)
        {
        case aafc::kGenericCollectorType:
          currentDatabase_->AddGenericCollector(static_cast<aaoc::GenericCollector&>(*br.Collector));
          break;
        case aafc::kNodeCollectorType:
          currentDatabase_->AddNodeCollector(static_cast<aaoc::NodeSetCollector&>(*br.Collector));
          break;
        case aafc::kElementCollectorType:
          currentDatabase_->AddElementCollector(static_cast<aaoc::ElementSetCollector&>(*br.Collector));
          break;
        default:
          assert(!_T("Unexpected enumeration value in ResultCollectorParser::Parse() method!"));
          break;
        }
			}
		}
		return br.Result;
	}

	// return our best effort
	return result;
}

void aapps::ResultCollectorParser::DoCloseContext( void )
{
  aapc::SymbolTable& st = GetParseContext().Symbols();
	bool fileNameChanged = false;
  bool shouldRequestNewStream = true;
	String extensionSupplied;
	String newExtension;
	String oldFileName;
	String newFileName;

  if (ShouldDiscardDatabase())
  { // discard database in failure or parsing skip request
    DestroyDatabase();
    return;
  }

	// trigger a warning if collector chain is empty
	if (!currentDatabase_->HasCollectors())
	{
		axis::String s = AXIS_WARN_MSG_EMPTY_OUTPUT_CHAIN;
		s += fileName_;
		GetParseContext().RegisterEvent(asmm::ErrorMessage(0x300564, s));
		return;
	}

  // create workbook
  aaow::ResultWorkbook& rw = workbookLocator_.BuildWorkbook(formatName_, formatArguments_);
  currentDatabase_->SetWorkbook(rw);
  if (!rw.SupportsAppendOperation() && appendToFile_)
  {
    axis::String s = AXIS_ERROR_MSG_OUTPUT_APPEND_ERROR;
    s.replace(_T("%1"), formatName_);
    GetParseContext().RegisterEvent(asmm::ErrorMessage(AXIS_ERROR_ID_OUTPUT_APPEND_ERROR, s));
    currentDatabase_->Destroy();
    currentDatabase_ = NULL;
    return;
  }
  rw.ToggleAppendOperation(appendToFile_);

  // make filename exclusive
  try
  {
    MarkFileNameAsUsed(rw.GetFormatIdentifier());
  }
  catch (axis::foundation::InvalidOperationException&)
  { // file already in use
    axis::String s = AXIS_ERROR_MSG_OUTPUT_FILE_CONFLICT;
    s.replace(_T("%1"), fileName_);
    GetParseContext().RegisterEvent(asmm::ErrorMessage(AXIS_ERROR_ID_OUTPUT_FILE_CONFLICT, s));

    currentDatabase_->Destroy();
    currentDatabase_ = NULL;
    return;
  }

  // drop database if block has already been processed
  if (st.IsSymbolDefined(GetAddedChainGlobalSymbolName(fileName_), 
                         aapc::SymbolTable::kOutputFileSettings))
  {
    DestroyDatabase();
    return;
  }
  rw.SetWorkbookOutputName(fileName_);
  resultBucket_.AddDatabase(*currentDatabase_);
  st.DefineOrRefreshSymbol(GetAddedChainGlobalSymbolName(fileName_), 
                           aapc::SymbolTable::kOutputFileSettings);

 	// tidy up
  currentDatabase_ = NULL;
	onErrorRecovering_ = false;
	onSkipBuild_ = false;
}

bool aapps::ResultCollectorParser::ShouldDiscardDatabase( void )
{
  return ((onSkipBuild_ || onErrorRecovering_));
}

void aapps::ResultCollectorParser::DestroyDatabase( void )
{
  currentDatabase_->Destroy();
  currentDatabase_ = NULL;
}

void aapps::ResultCollectorParser::MarkFileNameAsUsed( const axis::String& formatExtension )
{
  aapc::SymbolTable& st = GetParseContext().Symbols();
  bool fileNameChanged = false;
  bool shouldRequestNewStream = true;
  String extensionSupplied;
  String oldFileName = fileName_;
  String newFileName;

  // first, tidy supplied filename stripping any file extension supplied
  if (!asi::FileSystem::GetFileExtension(fileName_).empty())
  {
    extensionSupplied = asi::FileSystem::GetFileExtension(oldFileName);
    newFileName = asi::FileSystem::ReplaceFileExtension(oldFileName, formatExtension);
    fileNameChanged = true;

    String s = AXIS_INFO_MSG_RESULT_STRIPPED_EXTENSION;
    s = s.replace(_T("%1"), extensionSupplied)
         .replace(_T("%2"), oldFileName)
         .replace(_T("%3"), newFileName);
    GetParseContext().RegisterEvent(asmm::InfoMessage(0x100501, s));
  }
  else
  {
    newFileName = oldFileName + _T(".") + formatExtension;
  }

  if (!appendToFile_)
  {
    axis::String fileSymbolName = GetChainGlobalSymbolName(newFileName);
    if (st.IsSymbolCurrentRoundDefined(fileSymbolName, aapc::SymbolTable::kOutputFileSettings))
    {
      axis::String s = AXIS_ERROR_MSG_RESULT_DUPLICATED_STREAM;
      GetParseContext().RegisterEvent(asmm::ErrorMessage(0x300565, s));
      throw axis::foundation::InvalidOperationException();
    }
    else
    {
      st.DefineOrRefreshSymbol(fileSymbolName, aapc::SymbolTable::kOutputFileSettings);
    }
  }
  else
  {
    axis::String perStepSymbolName = GetChainPerStepSymbolName(newFileName, 
                                                               GetAnalysis().GetCurrentStepIndex());
    if (st.IsSymbolCurrentRoundDefined(perStepSymbolName, aapc::SymbolTable::kOutputFileSettings))
    {
      axis::String s = AXIS_ERROR_MSG_OUTPUT_APPEND_VIOLATION;
      GetParseContext().RegisterEvent(asmm::ErrorMessage(AXIS_ERROR_ID_OUTPUT_APPEND_VIOLATION, s));
      throw axis::foundation::InvalidOperationException();
    }
    else
    {
      st.DefineOrRefreshSymbol(perStepSymbolName, aapc::SymbolTable::kOutputFileSettings);
    }
  }
  fileName_ = newFileName;
}

axis::String aapps::ResultCollectorParser::GetChainGlobalSymbolName( const axis::String& fileName) const
{
  return _T("@") + fileName + _T("+");
}

axis::String aapps::ResultCollectorParser::GetChainPerStepSymbolName( const axis::String& fileName, 
                                                                     int stepIndex ) const
{
  return _T("@") + fileName + _T("@") + axis::String::int_parse(stepIndex);
}

axis::String aapps::ResultCollectorParser::GetAddedChainGlobalSymbolName( 
  const axis::String& fileName ) const
{
  return _T("@") + fileName + _T("++");
}
