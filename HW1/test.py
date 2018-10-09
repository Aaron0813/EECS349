import parse as ps
import ID3
# Test start
# classList = (ps.parse("house_votes_84.data"))
# test_data = dict(Outlook='Sunny', Temperature='Hot', Humidity='High', Wind='Weak')
test_data = dict(a=1, b=0)
classList = (ps.parse("my_data.data"))
# print(ID3.choose_most_feature(classList))
node = ID3.ID3(classList, 0)
ID3.evaluate(node,test_data)
# print((node.children['y']).label)
# print((node.children['n']).label)
