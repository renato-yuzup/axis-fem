#pragma once
#include "application/output/collectors/GenericCollector.hpp"
#include "SummaryType.hpp"
#include "domain/elements/FiniteElement.hpp"
#include "foundation/blas/DenseMatrix.hpp"
#include "foundation/Axis.CommonLibrary.hpp"

namespace axis { namespace application { namespace output { 
namespace collectors { namespace summarizers {

class AXISCOMMONLIBRARY_API SummaryElementMatrixCollector : 
  public axis::application::output::collectors::GenericCollector
{
public:
  SummaryElementMatrixCollector(const axis::String& targetSetName, 
    SummaryType summaryType, int rowCount, int colCount);
  virtual ~SummaryElementMatrixCollector(void);

  virtual void Collect(const axis::services::messaging::ResultMessage& message, 
    axis::application::output::recordsets::ResultRecordset& recordset, 
    const axis::domain::analyses::NumericalModel& numericalModel);
  virtual bool IsOfInterest( 
    const axis::services::messaging::ResultMessage& message ) const;
  virtual axis::String GetFriendlyDescription( void ) const;
private:
  virtual axis::String GetVariableName(void) const = 0;
  void StartCollect(axis::foundation::blas::DenseMatrix& matrix);
  virtual real CalculateMatrixNorm(
    const axis::services::messaging::ResultMessage& message, 
    const axis::domain::elements::FiniteElement& element) = 0;
  virtual const axis::foundation::blas::DenseMatrix& CollectMatrix(
    const axis::services::messaging::ResultMessage& message, 
    const axis::domain::elements::FiniteElement& element) = 0;

  axis::String targetSetName_;
  SummaryType summaryType_;
  int rowCount_, colCount_;
  id_type targetIdToCollect_;
  real bestNormValue_;
};

} } } } } // namespace axis::application::output::collectors::summarizers
