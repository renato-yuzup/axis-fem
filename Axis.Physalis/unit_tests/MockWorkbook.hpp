#pragma once
#if defined DEBUG || defined _DEBUG

#include "unit_tests.hpp"
#include "application/output/recordsets/ResultRecordset.hpp"
#include "application/output/workbooks/ResultWorkbook.hpp"

namespace axis { namespace unit_tests { namespace physalis {

class MockRecordset : public axis::application::output::recordsets::ResultRecordset
{
public:
  MockRecordset(int index);
  virtual void Destroy( void ) const;
  virtual bool IsInitialized( void ) const;
  virtual bool IsReady( void ) const;
  virtual void WriteData( bool data );
  virtual void WriteData( int data );
  virtual void WriteData( real data );
  virtual void WriteData( const axis::String& literal );
  virtual void WriteData( const axis::foundation::blas::DenseMatrix& data );
  virtual void WriteData( const axis::foundation::blas::SymmetricMatrix& data );
  virtual void WriteData( const axis::foundation::blas::LowerTriangularMatrix& data );
  virtual void WriteData( const axis::foundation::blas::UpperTriangularMatrix& data );
  virtual void WriteData( const axis::foundation::blas::ColumnVector& data );
  virtual void WriteData( const axis::foundation::blas::RowVector& data );
  int GetIndex(void) const;
private:
  int index_;
};

class MockWorkbook : public axis::application::output::workbooks::ResultWorkbook
{
public:
  virtual void Destroy( void ) const;
  virtual bool SupportsAppendOperation( void ) const;
  virtual axis::String GetFormatIdentifier( void ) const;
  virtual axis::String GetFormatTitle( void ) const;
  virtual axis::String GetShortDescription( void ) const;
  virtual bool SupportsNodeRecordset( void ) const;
  virtual bool SupportsElementRecordset( void ) const;
  virtual bool SupportsGenericRecordset( void ) const;
  virtual bool SupportsMainRecordset( void ) const;
  virtual axis::application::output::recordsets::ResultRecordset& DoCreateNodeRecordset( const axis::String& nodeSetName );
  virtual axis::application::output::recordsets::ResultRecordset& DoCreateElementRecordset( const axis::String& elementSetName );
  
  void SetNextRecordsetIndex(int index);
private:
  int index_;
};

} } } 

#endif
