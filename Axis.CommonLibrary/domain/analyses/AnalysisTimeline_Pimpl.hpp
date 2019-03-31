#pragma once
#include <set>
#include "AnalysisTimeline.hpp"
#include "SnapshotMark.hpp"
#include <list>
#include <boost/multi_index/random_access_index.hpp>
#include <boost/multi_index/indexed_by.hpp>
#include <boost/multi_index_container.hpp>

namespace axis { namespace domain { namespace analyses {

class AnalysisTimeline::Pimpl
{
public:

// stores info of the last snapshot mark
size_type nextSnapshotIndex;
real currentSnapshotMarkTime;
bool hasCrossedSnapshotMark;

// the snapshot marks
typedef boost::multi_index::random_access<> numbered_index;
typedef boost::multi_index::indexed_by<numbered_index> index;
typedef boost::multi_index::multi_index_container<SnapshotMark *, index> collection;
collection snapshotMarks;
};

} } } // namespace axis::domain::analyses
