from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
  'torch',
  'transformers',
  'datasets'
]

setup(
  name='anime_chatbot',
  version='0.1',
  author='WaifuAI',
  author_email='waifuai@users.noreply.github.com',
  url='https://github.com/waifuai/anime-subtitle-chatbot-trax',
  install_requires=REQUIRED_PACKAGES,
  extras_require={
      'test': ['pytest']
  },
  packages=find_packages(),
  include_package_data=True,
  description='Anime Chatbot Problem',
  requires=[] # Note: 'requires' is deprecated, install_requires is preferred
)
