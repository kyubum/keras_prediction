from pandas import read_csv
from datetime import datetime

dataset = read_csv('/Users/kyubum/PycharmProjects/PRACTICE/prac/flamingo.csv')
dataset.drop(1,0)
dataset.columns = ['date', '요일', '영업매장', '총매출', '총할인', '실매출', '가액', '부가세','영수건수','영수단가','고객수',
                   '객단가','남','여','봉사료','에누리','결제합계','단순현금','현금영수','신용카드','외상','상품권','식권',
                   '회원포인트','제휴카드','사원카드','모바일쿠폰','캐시비','일반','점유율(%)','포장','점유율(%)','배달','점유율(%)',
                   '일반','서비스','제휴','쿠폰','회원','식권','포장','사후환급액','즉시환급액','환급수수료']

dataset.drop('요일', axis=1, inplace=True)
dataset.drop('영업매장', axis=1, inplace=True)
dataset.drop('실매출', axis=1, inplace=True)
dataset.drop('가액', axis=1, inplace=True)
dataset.drop('부가세', axis=1, inplace=True)
dataset.drop('영수단가', axis=1, inplace=True)
dataset.drop('고객수', axis=1, inplace=True)
dataset.drop('객단가', axis=1, inplace=True)
dataset.drop('남', axis=1, inplace=True)
dataset.drop('여', axis=1, inplace=True)
dataset.drop('봉사료', axis=1, inplace=True)
dataset.drop('에누리', axis=1, inplace=True)
dataset.drop('결제합계', axis=1, inplace=True)
dataset.drop('단순현금', axis=1, inplace=True)
dataset.drop('현금영수', axis=1, inplace=True)
dataset.drop('신용카드', axis=1, inplace=True)
dataset.drop('외상', axis=1, inplace=True)
dataset.drop('상품권', axis=1, inplace=True)
dataset.drop('식권', axis=1, inplace=True)
dataset.drop('회원포인트', axis=1, inplace=True)
dataset.drop('제휴카드', axis=1, inplace=True)
dataset.drop('사원카드', axis=1, inplace=True)
dataset.drop('모바일쿠폰', axis=1, inplace=True)
dataset.drop('캐시비', axis=1, inplace=True)
dataset.drop('일반', axis=1, inplace=True)
dataset.drop('점유율(%)', axis=1, inplace=True)
dataset.drop('제휴', axis=1, inplace=True)
dataset.drop('쿠폰', axis=1, inplace=True)
dataset.drop('회원', axis=1, inplace=True)
dataset.drop('식권', axis=1, inplace=True)
dataset.drop('포장', axis=1, inplace=True)
dataset.drop('사후환급액', axis=1, inplace=True)
dataset.drop('즉시환급액', axis=1, inplace=True)
dataset.drop('환급수수료', axis=1, inplace=True)
dataset.drop('배달', axis=1, inplace=True)

dataset.index.name = 'date'
dataset.drop('date', axis=1, inplace=True)
dataset.to_csv('flamingo2.csv')
