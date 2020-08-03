# ninjasat (NinjaSat 運用シミュレーター)
(Only the NinjaSat team internal usage is assumed at this moment.; 現時点では榎戸の個人的な理解のために作成している)

NinjaSat の衛星運用の様子を把握するために作成しているライブラリ群。cubesat 以下にある cubesat.py の基本的なライブラリを用意しており、cubesat/cli/ 以下にあるのが、コマンドラインで実行することを想定したコマンド。

## Required libraries

標準的なライブラリの他に、以下のライブラリが必要になる。いずれも pip install xxx でインストールできるはず。ただし、cartopy を動かすには、予め

```
brew install proj
brew install geos
```

の 2つのライブラリを入れておく必要がある。

- pyorbital: https://pypi.org/project/orbit-predictor/
- cartopy: 

```
%> pip install pyorbital 
```
などとする。

## How to install 

```
%> git clone https://github.com/tenoto/ninjasat.git
%> source setenv/setenv.bashrc    
```

## Directory structure 



## How to use 

```
tests/plot_ninjasat_orbit.sh 
```

## Files in the "data" directory

- ninjasat_setup.yaml: NinjaSat の想定パラメータを一式格納
- iss.tle: ISS の TLE 要素ファイル
- lookuptable_hvoff.csv: GMC の HV を off する想定の領域を指定

## Change Log 

- 2020-08-03: first version was uplaoded.