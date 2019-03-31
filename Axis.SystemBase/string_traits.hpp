#pragma once
#include <tchar.h>
#include <xstring>
#include <iosfwd>
#include "AxisString.hpp"

template <> struct std::iterator_traits<axis::AsciiString::iterator> {
  typedef axis::AsciiString::iterator::difference_type difference_type;
  typedef axis::AsciiString::iterator::value_type value_type;
  typedef axis::AsciiString::iterator::pointer pointer;
  typedef axis::AsciiString::iterator::reference reference;
  typedef std::bidirectional_iterator_tag iterator_category;
};
template <> struct std::iterator_traits<axis::UnicodeString::iterator> {
  typedef axis::UnicodeString::iterator::difference_type difference_type;
  typedef axis::UnicodeString::iterator::value_type value_type;
  typedef axis::UnicodeString::iterator::pointer pointer;
  typedef axis::UnicodeString::iterator::reference reference;
  typedef std::bidirectional_iterator_tag iterator_category;
};

template <> struct std::iterator_traits<axis::AsciiString::reverse_iterator> {
  typedef axis::AsciiString::reverse_iterator::difference_type difference_type;
  typedef axis::AsciiString::reverse_iterator::value_type value_type;
  typedef axis::AsciiString::reverse_iterator::pointer pointer;
  typedef axis::AsciiString::reverse_iterator::reference reference;
  typedef std::bidirectional_iterator_tag iterator_category;
};
template <> struct std::iterator_traits<axis::UnicodeString::reverse_iterator> {
  typedef axis::UnicodeString::reverse_iterator::difference_type difference_type;
  typedef axis::UnicodeString::reverse_iterator::value_type value_type;
  typedef axis::UnicodeString::reverse_iterator::pointer pointer;
  typedef axis::UnicodeString::reverse_iterator::reference reference;
  typedef std::bidirectional_iterator_tag iterator_category;
};

namespace axis {

template<class traits>
std::basic_istream<wchar_t, traits>& getline(std::basic_istream<wchar_t, traits>& istr, axis::UnicodeString& str)
{
  str.clear();
  wchar_t c = istr.get();
  while (c != istr.widen('\n') && c != istr.widen('\r') && !istr.eof())
  {
    str += c;
    c = istr.get();
  }
  return istr;
}
template<class traits>
std::basic_istream<char, traits>& getline(std::basic_istream<char, traits>& istr, axis::AsciiString& str)
{
  str.clear();
  char c = istr.get();
  while (c != '\n' && c != '\r' && !istr.eof())
  {
    str += c;
    c = istr.get();
  }
  return istr;
}

template <class traits>
std::basic_istream<char,traits>& operator >>(std::basic_istream<char,traits>& is, axis::AsciiString& str )
{
  str.clear();
  char c = is.get();
  while(c != ' ' && c != '\t' && c != '\n' && c != '\r' && c != '\0' && !is.eof())
  {
    str.append(c);
    c = is.get();
  }
  return is;
}
template <class traits>
std::basic_istream<wchar_t,traits>& operator >>(std::basic_istream<wchar_t,traits>& is, axis::UnicodeString& str )
{
  str.clear();
  wchar_t c = is.get();
  while(c != ' ' && c != '\t' && c != '\n' && c != '\r' && c != '\0' && !is.eof())
  {
    str.append((char_type)c);
    c = is.get();
  }
  return is;
}
template <class traits>
std::basic_istream<char,traits>& operator >>(std::basic_istream<char,traits>& is, axis::UnicodeString& str )
{
  str.clear();
  char c = is.get();
  while(c != ' ' && c != '\t' && c != '\n' && c != '\r' && c != '\0' && !is.eof())
  {
    str.append((wchar_t)c);
    if (is.peek() == std::char_traits<char>::eof())
    {
      break;
    }
    c = is.get();
  }
  return is;
}

template<class traits>
std::basic_ostream<char,traits>& operator<<(std::basic_ostream<char,traits>& os, const axis::AsciiString& str )
{
  os << str.data();
  return os;
}
template<class traits>
std::basic_ostream<wchar_t,traits>& operator<<(std::basic_ostream<wchar_t,traits>& os, const axis::UnicodeString& str )
{
  os << str.data();
  return os;
}
template<class traits>
std::basic_ostream<char,traits>& operator<<(std::basic_ostream<char,traits>& os, const axis::UnicodeString& str )
{
  size_t n = str.size();
  wchar_t *ptr = str.data();
  for (size_t i = 0; i < n; )
  {
    os << (ptr[i] < 255? ptr[i] : 255);
  }
  return os;
}

} // namespace axis
