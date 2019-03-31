#ifndef AXIS_SYSTEM_BASE_AXIS_NOCOPY_HPP_
#define AXIS_SYSTEM_BASE_AXIS_NOCOPY_HPP_

// A convenient macro to disable copy constructor and
// copy assignment operator; should be used in a private
// declaration block of a class
#define DISALLOW_COPY_AND_ASSIGN(TypeName)      TypeName(const TypeName&); \
                                                TypeName& operator =(const TypeName&)

#endif // AXIS_SYSTEM_BASE_AXIS_NOCOPY_HPP