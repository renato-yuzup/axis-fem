#pragma once
#include "application/output/recordsets/ResultRecordset.hpp"
#include "services/io/StreamWriter.hpp"

namespace axis { namespace application { 

namespace jobs {
class WorkFolder;
} // namespace axis::application::jobs

namespace output { namespace recordsets {

class TextReportRecordset : public ResultRecordset
{
public:
  TextReportRecordset(const axis::String& outputFileName, 
    axis::application::jobs::WorkFolder& workFolder);
  ~TextReportRecordset(void);

  virtual void Destroy( void ) const;
  virtual void OpenRecordset( const axis::String& entitySet );
  virtual void CloseRecordset( void );
  virtual void BeginSnapshot( const axis::domain::analyses::AnalysisInfo& analysisInfo );
  virtual bool IsInitialized( void ) const;
  virtual bool IsReady( void ) const;
  virtual void WriteData(int data);
  virtual void WriteData(real data);
  virtual void WriteData(bool data);
  virtual void WriteData( const axis::String& literal );
  virtual void WriteData( const axis::foundation::blas::DenseMatrix& data );
  virtual void WriteData( const axis::foundation::blas::SymmetricMatrix& data );
  virtual void WriteData( const axis::foundation::blas::LowerTriangularMatrix& data );
  virtual void WriteData( const axis::foundation::blas::UpperTriangularMatrix& data );
  virtual void WriteData( const axis::foundation::blas::ColumnVector& data );
  virtual void WriteData( const axis::foundation::blas::RowVector& data );

  void RawWriteLine( const axis::String& line );
  void RawWrite( const axis::String& line );
  void ForcedWriteLine( const axis::String& line );
  void ForcedWrite( const axis::String& line );

  virtual void EndGenericRecord( const axis::services::messaging::ResultMessage& message, 
                                  const axis::domain::analyses::NumericalModel& numericalModel );
private:
  axis::services::io::StreamWriter& writer_;
  bool sleeping_;
};

} } } } // namespace axis::application::output::recordsets
