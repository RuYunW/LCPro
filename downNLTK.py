import nltk
import ssl

# 取消ssl认证
ssl._create_default_https_context = ssl._create_unverified_context

# download nltk data package
nltk.download()