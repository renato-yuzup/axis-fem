#include "StaticAnalysisInfo.hpp"

namespace ada = axis::domain::analyses;

ada::StaticAnalysisInfo::StaticAnalysisInfo(void)
{
  // nothing to do here
}

ada::StaticAnalysisInfo::~StaticAnalysisInfo(void)
{
  // nothing to do here
}

void ada::StaticAnalysisInfo::Destroy( void ) const
{
  delete this;
}

ada::AnalysisInfo& ada::StaticAnalysisInfo::Clone( void ) const
{
  return *new StaticAnalysisInfo();
}


ada::AnalysisInfo::AnalysisType ada::StaticAnalysisInfo::GetAnalysisType( void ) const
{
  return StaticAnalysis;
}