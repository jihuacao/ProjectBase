#ifndef PROJECT_BASE_MULTI_LANGUAGE_H
#define PROJECT_BASE_MULTI_LANGUAGE_H
#include <string>
#include <ProjectBase/multi_language/Define.hpp>
class PROJECT_BASE_MULTI_LANGUAGE_SYMBOL Languages{
public:
    Languages();
    Languages(const Languages& other);
    Languages(Languages&& right);
public:
};

class PROJECT_BASE_MULTI_LANGUAGE_SYMBOL Language{
public:
    Language();
    Language(const Language&);
    Language(Language&& right);
};

class PROJECT_BASE_MULTI_LANGUAGE_SYMBOL LanguageIndex{
public:
    LanguageIndex();
    LanguageIndex(const std::string& index);
    LanguageIndex(const LanguageIndex& other);
    LanguageIndex(LanguageIndex&& right);
public:
    operator std::string() const;
};

class multi_language;
class PROJECT_BASE_MULTI_LANGUAGE_SYMBOL MultiLanguage
{
public:
    MultiLanguage();
    MultiLanguage(MultiLanguage &&) = delete;
    MultiLanguage(const MultiLanguage &) = delete;
    MultiLanguage &operator=(MultiLanguage &&) = delete;
    MultiLanguage &operator=(const MultiLanguage &) = delete;
    ~MultiLanguage();
public:
    void SetLanguage(const Language& language);
    void SetLanguageFold(const std::string& pth);
public:
    const Languages& SupportedLanguages() const;
    const std::string Extract(const LanguageIndex& index);
    
public:
    multi_language* _impl;
};
#else
#endif // !1_H_MULTI_LANGUAG