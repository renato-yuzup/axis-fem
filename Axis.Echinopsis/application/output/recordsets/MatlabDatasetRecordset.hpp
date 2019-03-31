#pragma once
#include "application/output/recordsets/ResultRecordset.hpp"
#include "services/io/StreamWriter.hpp"

namespace axis { namespace application { 

namespace jobs {
class WorkFolder;
} // namespace axis::application::jobs

namespace output { namespace recordsets {

class MatlabDatasetRecordset : public ResultRecordset
{
public:
  MatlabDatasetRecordset(const axis::String& outputFileName, 
                         const axis::String& variableName, 
                         axis::application::jobs::WorkFolder& workFolder);
  ~MatlabDatasetRecordset(void);

  virtual void Destroy( void ) const;
  virtual void OpenRecordset( const axis::String& entitySet );
  virtual void CloseRecordset( void );
  virtual void BeginAnalysisStep( const axis::String& stepName, int stepIndex, const axis::services::diagnostics::information::SolverCapabilities& solverCaps );
  virtual void EndAnalysisStep( void );
  virtual void BeginSnapshot( const axis::domain::analyses::AnalysisInfo& analysisInfo );
  virtual void EndSnapshot( const axis::domain::analyses::AnalysisInfo& analysisInfo );
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
private:
  axis::String pendingLine_;
  axis::String variableName_;
  axis::services::io::StreamWriter& writer_;
};

} } } } // namespace axis::application::output::recordsets
