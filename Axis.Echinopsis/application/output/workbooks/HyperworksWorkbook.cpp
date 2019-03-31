#include "stdafx.h"
#include "HyperworksWorkbook.hpp"
#include "application/jobs/AnalysisStepInformation.hpp"
#include "application/jobs/WorkFolder.hpp"
#include "application/output/recordsets/HyperworksNodeRecordset.hpp"
#include "application/output/recordsets/HyperworksElementRecordset.hpp"
#include "foundation/IOException.hpp"
#include "../../../../Axis.CommonLibrary/Services/IO/StreamReader.hpp"

namespace aaj = axis::application::jobs;
namespace aaow = axis::application::output::workbooks;
namespace aaor = axis::application::output::recordsets;
namespace asio = axis::services::io;

namespace {
  inline void CopyDataFromTempFile(asio::StreamWriter& writer, asio::StreamReader& reader)
  {
    axis::String line;
    line.reserve(1024*1024);
    while (!reader.IsEOF())
    {
      reader.ReadLine(line);
      writer.WriteLine(line);
    }
  }
}

aaow::HyperworksWorkbook::HyperworksWorkbook( void )
{
  writer_ = NULL;
  workFolder_ = NULL;
}

aaow::HyperworksWorkbook::~HyperworksWorkbook( void )
{
  if (writer_ != NULL)
  {
    if (writer_->IsOpen()) writer_->Close();
    writer_->Destroy();
    writer_ = NULL;
  }
}

void aaow::HyperworksWorkbook::Destroy( void ) const
{
  delete this;
}

bool aaow::HyperworksWorkbook::SupportsAppendOperation( void ) const
{
  return true;
}

axis::String aaow::HyperworksWorkbook::GetFormatIdentifier( void ) const
{
  return _T("hwascii");
}

axis::String aaow::HyperworksWorkbook::GetFormatTitle( void ) const
{
  return _T("Hyperworks(R) ASCII Format");
}

axis::String aaow::HyperworksWorkbook::GetShortDescription( void ) const
{
  return _T("Output format for use in Altair(R) Hyperworks(R) program suite.");
}

bool aaow::HyperworksWorkbook::SupportsNodeRecordset( void ) const
{
  return true;
}

bool aaow::HyperworksWorkbook::SupportsElementRecordset( void ) const
{
  return true;
}

bool aaow::HyperworksWorkbook::SupportsGenericRecordset( void ) const
{
  return false;
}

bool aaow::HyperworksWorkbook::SupportsMainRecordset( void ) const
{
  return false;
}

aaor::ResultRecordset& aaow::HyperworksWorkbook::DoCreateNodeRecordset( const axis::String& )
{
  return *new aaor::HyperworksNodeRecordset();
}

aaor::ResultRecordset& aaow::HyperworksWorkbook::DoCreateElementRecordset( const axis::String& )
{
  return *new aaor::HyperworksElementRecordset();
}

bool aaow::HyperworksWorkbook::IsReady( void ) const
{
  if (writer_ == NULL) return false;
  return writer_->IsOpen();
}

void aaow::HyperworksWorkbook::DoAfterInit( aaj::WorkFolder& workFolder )
{
  // open main file
  writer_ = &workFolder.GetOrCreateWorkFile(GetWorkbookOutputName());
  workFolder_ = &workFolder;
}

void aaow::HyperworksWorkbook::DoAfterOpen( const aaj::AnalysisStepInformation& stepInfo )
{
  writer_->Open(IsAppendOperation()? asio::StreamWriter::kAppend : asio::StreamWriter::kOverwrite,
                asio::StreamWriter::kExclusiveMode);
  if (!writer_->IsOpen())
  {
    throw axis::foundation::IOException();
  }
  writer_->WriteLine(_T("ALTAIR ASCII FILE"));
  String titleStatement = _T("$TITLE = ");
  String subcaseName = stepInfo.GetStepName();
  subcaseName.trim();
  titleStatement += stepInfo.GetJobTitle();
  writer_->WriteLine(titleStatement);
  String subcaseStatement = _T("$SUBCASE = ");
  subcaseStatement += String::int_parse(stepInfo.GetStepIndex() + 1);
  subcaseStatement += _T("    ");
  subcaseStatement += subcaseName.empty()? _T("<UNTITLED>") : stepInfo.GetStepName();
  writer_->WriteLine(subcaseStatement);
}

void aaow::HyperworksWorkbook::DoBeforeClose( void )
{
  // collect temporary filenames
  int nodeRecCount = GetNodeRecordsetCount();
  for (int i = 0; i < nodeRecCount; ++i)
  {
    aaor::HyperworksNodeRecordset& r = 
        static_cast<aaor::HyperworksNodeRecordset&>(GetNodeRecordset(i));
    tempFiles_.push_back(r.GetTempFileLocation());
  }
  int elementRecCount = GetElementRecordsetCount();
  for (int i = 0; i < elementRecCount; ++i)
  {
    aaor::HyperworksElementRecordset& r = 
        static_cast<aaor::HyperworksElementRecordset&>(GetElementRecordset(i));
    tempFiles_.push_back(r.GetTempFileLocation());
  }
}

void aaow::HyperworksWorkbook::DoAfterClose( void )
{
  tempfile_list::iterator end = tempFiles_.end();
  for (tempfile_list::iterator it = tempFiles_.begin(); it != end; ++it)
  {
    axis::String filename = *it;
    asio::StreamReader& reader = workFolder_->OpenTempFileForRead(filename);
    CopyDataFromTempFile(*writer_, reader);
    reader.Close();
    delete &reader;
  }
  writer_->Close();
}
