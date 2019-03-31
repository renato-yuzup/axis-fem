#include "Sketchbook.hpp"
#include "Sketchbook_Pimpl.hpp"
#include "foundation/ArgumentException.hpp"

namespace aapc = axis::application::parsing::core;

aapc::Sketchbook::Sketchbook( void )
{
	pimpl_ = new Pimpl();
}

aapc::Sketchbook::~Sketchbook( void )
{
  // delete all sections defined	
  Pimpl::section_set::iterator end = pimpl_->sections.end();
  for (Pimpl::section_set::iterator it = pimpl_->sections.begin(); it != end; ++it)
  {
    delete it->second;
  }
	delete pimpl_;
}

bool aapc::Sketchbook::HasSectionDefined( const axis::String& setId ) const
{
  return pimpl_->sections.find(setId) != pimpl_->sections.end();
}

void aapc::Sketchbook::AddSection( const axis::String& setId, const SectionDefinition& sectionDefinition )
{
  if (HasSectionDefined(setId))
  {
    throw axis::foundation::ArgumentException();
  }
  pimpl_->sections[setId] = &sectionDefinition;
}

const aapc::SectionDefinition& aapc::Sketchbook::RemoveSection( const axis::String& setId )
{
  if (!HasSectionDefined(setId))
  {
    throw axis::foundation::ArgumentException();
  }
  const SectionDefinition *section = pimpl_->sections[setId];
  pimpl_->sections.erase(setId);

  return *section;
}

unsigned int aapc::Sketchbook::SectionsCount( void ) const
{
  return (unsigned int)pimpl_->sections.size();
}

const aapc::SectionDefinition& aapc::Sketchbook::GetSection( const axis::String& setId ) const
{
  if (!HasSectionDefined(setId))
  {
    throw axis::foundation::ArgumentException();
  }
  return *pimpl_->sections.at(setId);
}

void aapc::Sketchbook::Destroy( void ) const
{
  delete this;
}

