#pragma once
#include "EchinopsisPluginLoader.hpp"
#include "application/factories/workbooks/MatlabDatasetWorkbookFactory.hpp"
#include "application/factories/collectors/MatlabDatasetCollectorFactory.hpp"
#include "application/factories/workbooks/TextReportWorkbookFactory.hpp"
#include "application/factories/collectors/TextReportNodeCollectorFactory.hpp"
#include "application/factories/collectors/TextReportElementCollectorFactory.hpp"
#include "application/factories/workbooks/HyperworksWorkbookFactory.hpp"
#include "application/factories/collectors/HyperworksNodeCollectorFactory.hpp"
#include "application/factories/collectors/HyperworksElementCollectorFactory.hpp"

namespace axis { namespace services { namespace management {

class EchinopsisPluginLoader::Pimpl
{
public:
  Pimpl(void);
  ~Pimpl(void);
  axis::application::factories::workbooks::MatlabDatasetWorkbookFactory *MatlabFactory;
  axis::application::factories::collectors::MatlabDatasetCollectorFactory *MatlabCollectorFactory;
  axis::application::factories::workbooks::TextReportWorkbookFactory *ReportFactory;
  axis::application::factories::collectors::TextReportNodeCollectorFactory *ReportNodeCollectorFactory;
  axis::application::factories::collectors::TextReportElementCollectorFactory *ReportElementCollectorFactory;
  axis::application::factories::workbooks::HyperworksWorkbookFactory *HyperworksFactory;
  axis::application::factories::collectors::HyperworksNodeCollectorFactory *HyperworksNodeCollectorFactory;
  axis::application::factories::collectors::HyperworksElementCollectorFactory *HyperworksElementCollectorFactory;
};

} } } // namespace axis::services::management
