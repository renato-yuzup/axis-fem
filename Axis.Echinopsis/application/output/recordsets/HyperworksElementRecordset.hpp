#pragma once
#include "application/output/recordsets/ResultRecordset.hpp"
#include "services/io/StreamWriter.hpp"

namespace axis { namespace application { 

namespace jobs {
class WorkFolder;
} // namespace axis::application::jobs

namespace output { namespace recordsets {

class HyperworksElementRecordset : public ResultRecordset
{
public:
  HyperworksElementRecordset(void);
  ~HyperworksElementRecordset(void);
  virtual void Destroy( void ) const;

  virtual void Init( axis::application::jobs::WorkFolder& workFolder );

  virtual void OpenRecordset( const axis::String& entitySet );
  virtual void CloseRecordset( void );

  virtual void BeginCreateField( void );
  virtual void CreateField( const axis::String& fieldName, FieldType fieldType );
  virtual void CreateMatrixField( const axis::String& fieldName, int rows, int columns );
  virtual void CreateVectorField( const axis::String& fieldName, int length );
  virtual void EndCreateField( void );

  virtual void BeginAnalysisStep( 
      const axis::String& stepName, int stepIndex, 
      const axis::services::diagnostics::information::SolverCapabilities& solverCaps );
  virtual void EndAnalysisStep( void );

  virtual void BeginSnapshot( const axis::domain::analyses::AnalysisInfo& analysisInfo );

  virtual void BeginElementRecord( const axis::services::messaging::ResultMessage& message, 
                                    const axis::domain::elements::FiniteElement& element );
  virtual void EndElementRecord( const axis::services::messaging::ResultMessage& message, 
                                  const axis::domain::elements::FiniteElement& element );

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

  virtual axis::String GetTempFileLocation(void) const;
private:
  void WriteFieldName(const axis::String& fieldName);

  axis::String internalBuffer_;
  axis::String resultTypeStatement_;
  axis::services::io::StreamWriter *writer_;
};

} } } } // namespace axis::application::output::recordsets
