import numpy as np


def compute_feature_distances(features1, features2):
    """
    This function computes a list of distances from every feature in one array
    to every feature in another.
    Args:
    - features1: A numpy array of shape (n,feat_dim) representing one set of
      features, where feat_dim denotes the feature dimensionality
    - features2: A numpy array of shape (m,feat_dim) representing a second set
      features (m not necessarily equal to n)

    Returns:
    - dists: A numpy array of shape (n,m) which holds the distances from each
      feature in features1 to each feature in features2
    """

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################
    dists = np.empty((features1.shape[0], features2.shape[0]))
    for m in range(features2.shape[0]):
      features2_row = features2[m,:]
      features2_row_tile = np.tile(features2_row, (features1.shape[0],1))
      dist = np.linalg.norm(features1 - features2_row_tile, axis = 1)
      dists[:, m] = dist

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dists


def match_features(features1, features2, x1, y1, x2, y2):
    """
    This function does not need to be symmetric (e.g. it can produce
    different numbers of matches depending on the order of the arguments).

    To start with, simply implement the "ratio test", equation 4.18 in
    section 4.1.3 of Szeliski. There are a lot of repetitive features in
    these images, and all of their descriptors will look similar. The
    ratio test helps us resolve this issue (also see Figure 11 of David
    Lowe's IJCV paper).

    You should call `compute_feature_distances()` in this function, and then
    process the output.

    Args:
    - features1: A numpy array of shape (n,feat_dim) representing one set of
      features, where feat_dim denotes the feature dimensionality
    - features2: A numpy array of shape (m,feat_dim) representing a second
      set of features (m not necessarily equal to n)
    - x1: A numpy array of shape (n,) containing the x-locations of features1
    - y1: A numpy array of shape (n,) containing the y-locations of features1
    - x2: A numpy array of shape (m,) containing the x-locations of features2
    - y2: A numpy array of shape (m,) containing the y-locations of features2

    Returns:
    - matches: A numpy array of shape (k,2), where k is the number of matches.
      The first column is an index in features1, and the second column is an
      index in features2
    - confidences: A numpy array of shape (k,) with the real valued confidence
      for every match

    'matches' and 'confidences' can be empty e.g. (0x2) and (0x1)
    """

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################
    dists = compute_feature_distances(features1, features2)
    sort_by_dist_idx = np.argsort(dists, axis=1)[:, :2]
    
    feature_row = np.arange(dists.shape[0])
    feature_col_1 = sort_by_dist_idx[:, 0]
    feature_col_2 = sort_by_dist_idx[:, 1]
    # calculate nndr
    nndr = dists[feature_row, feature_col_1] / dists[feature_row, feature_col_2]
    # sort nndr
    nndr_index = np.argsort(nndr)
    nndr = nndr[nndr_index]
    # get index with nndr above threshold of 0.8
    # magic number :\
    valid_index = nndr <= 0.8
    # make 2d array of (feature1, feature2)
    matches = np.vstack((feature_row, feature_col_1)).transpose()
    # sort by confidence
    matches = matches[nndr_index]
    # remove values above threshold
    matches = matches[valid_index]
    
    confidences = nndr[valid_index]

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return matches, confidences