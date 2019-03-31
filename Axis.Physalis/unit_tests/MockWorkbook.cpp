#if defined DEBUG || defined _DEBUG

#include "MockWorkbook.hpp"


axis::unit_tests::physalis::MockRecordset::MockRecordset( int index )
{
  index_ = index;
}

void axis::unit_tests::physalis::MockRecordset::Destroy( void ) const
{
  delete this;
}

bool axis::unit_tests::physalis::MockRecordset::IsInitialized( void ) const
{
  return true;
}

bool axis::unit_tests::physalis::MockRecordset::IsReady( void ) const
{
  return true;
}

void axis::unit_tests::physalis::MockRecordset::WriteData( bool )
{
  // nothing to do here
}

void axis::unit_tests::physalis::MockRecordset::WriteData( int )
{
  // nothing to do here
}

void axis::unit_tests::physalis::MockRecordset::WriteData( real )
{
  // nothing to do here
}

void axis::unit_tests::physalis::MockRecordset::WriteData( const axis::String& )
{
  // nothing to do here
}

void axis::unit_tests::physalis::MockRecordset::WriteData( const axis::foundation::blas::DenseMatrix& )
{
  // nothing to do here
}

void axis::unit_tests::physalis::MockRecordset::WriteData( const axis::foundation::blas::SymmetricMatrix& )
{
  // nothing to do here
}

void axis::unit_tests::physalis::MockRecordset::WriteData( const axis::foundation::blas::LowerTriangularMatrix& )
{
  // nothing to do here
}

void axis::unit_tests::physalis::MockRecordset::WriteData( const axis::foundation::blas::UpperTriangularMatrix& )
{
  // nothing to do here
}

void axis::unit_tests::physalis::MockRecordset::WriteData( const axis::foundation::blas::ColumnVector& )
{
  // nothing to do here
}

void axis::unit_tests::physalis::MockRecordset::WriteData( const axis::foundation::blas::RowVector& )
{
  // nothing to do here
}

int axis::unit_tests::physalis::MockRecordset::GetIndex( void ) const
{
  return index_;
}







void axis::unit_tests::physalis::MockWorkbook::Destroy( void ) const
{
  delete this;
}

bool axis::unit_tests::physalis::MockWorkbook::SupportsAppendOperation( void ) const
{
  return true;
}

axis::String axis::unit_tests::physalis::MockWorkbook::GetFormatIdentifier( void ) const
{
  return _T("test");
}

axis::String axis::unit_tests::physalis::MockWorkbook::GetFormatTitle( void ) const
{
  return _T("Workbook Test");
}

axis::String axis::unit_tests::physalis::MockWorkbook::GetShortDescription( void ) const
{
  return _T("");
}

bool axis::unit_tests::physalis::MockWorkbook::SupportsNodeRecordset( void ) const
{
  return true;
}

bool axis::unit_tests::physalis::MockWorkbook::SupportsElementRecordset( void ) const
{
  return true;
}

bool axis::unit_tests::physalis::MockWorkbook::SupportsGenericRecordset( void ) const
{
  return false;
}

bool axis::unit_tests::physalis::MockWorkbook::SupportsMainRecordset( void ) const
{
  return false;
}

axis::application::output::recordsets::ResultRecordset& axis::unit_tests::physalis::MockWorkbook::DoCreateNodeRecordset( const axis::String& nodeSetName )
{
  return *new MockRecordset(index_);
}

axis::application::output::recordsets::ResultRecordset& axis::unit_tests::physalis::MockWorkbook::DoCreateElementRecordset( const axis::String& elementSetName )
{
  return *new MockRecordset(index_);
}

void axis::unit_tests::physalis::MockWorkbook::SetNextRecordsetIndex( int index )
{
  index_ = index;
}

#endif