__author__ = 'QiYE'
def creat_idx_pair(type=0):
    if type == 0:
        print 'whole hand constraint'
        idx_pair =[]
        # jnt_idx = [2,6,10,14,18]
        jnt_idx = [1,5,9,13,17]
        for i in jnt_idx:
            idx_pair.append((i,0))
        for i in xrange(len(jnt_idx)-1):
            idx_pair.append((jnt_idx[i+1],jnt_idx[i]))
        for i in jnt_idx:
            for j in xrange(i,i+3,1):
                idx_pair.append((j+1,j))

        print idx_pair
        return idx_pair
    if type ==1:
        print 'base and twist'
        idx_pair =[]
        # jnt_idx = [2,6,10,14,18]
        jnt_idx = range(1,6,1)
        for i in jnt_idx:
            idx_pair.append((i,0))
        for i in xrange(len(jnt_idx)-1):
            idx_pair.append((jnt_idx[i+1],jnt_idx[i]))
        print idx_pair
        return idx_pair