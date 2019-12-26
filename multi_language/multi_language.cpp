#include <ProjectBase/multi_language/multi_language.hpp>
//Languages
Languages::Languages(){}
Languages::Languages(const Languages& other){}
Languages::Languages(Languages&& right){}

//Language
Language::Language(){}
Language::Language(const Language& other){}
Language::Language(Language&& right){}

//LanguageIndex
LanguageIndex::LanguageIndex(){}
LanguageIndex::LanguageIndex(const LanguageIndex& other){}
LanguageIndex::LanguageIndex(LanguageIndex&& right){}

// MultiLanguage
MultiLanguage::MultiLanguage(){}
MultiLanguage::~MultiLanguage(){}
void MultiLanguage::SetLanguageFold(const std::string& pth){}
void MultiLanguage::SetLanguage(const Language& name){}
const std::string MultiLanguage::Extract(const LanguageIndex& index){return std::string(index);}