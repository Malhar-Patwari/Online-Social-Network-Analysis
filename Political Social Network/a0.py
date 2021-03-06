from collections import Counter
import matplotlib.pyplot as plt
import networkx as nx
import sys
import time
from TwitterAPI import TwitterAPI


consumer_key = 
consumer_secret = 
access_token = 
access_token_secret = 


# This method is done for you.
def get_twitter():
    """ Construct an instance of TwitterAPI using the tokens you entered above.
    Returns:
      An instance of TwitterAPI.
    """
    return TwitterAPI(consumer_key, consumer_secret, access_token, access_token_secret)


def read_screen_names(filename):
    
    fname = open(filename)
    rows = fname.read()        
    return rows.split('\n')
    


# I've provided the method below to handle Twitter's rate limiting.
# You should call this method whenever you need to access the Twitter API.
def robust_request(twitter, resource, params, max_tries=5):
    """ If a Twitter request fails, sleep for 15 minutes.
    Do this at most max_tries times before quitting.
    Args:
      twitter .... A TwitterAPI object.
      resource ... A resource string to request; e.g., "friends/ids"
      params ..... A parameter
       dict for the request, e.g., to specify
                   parameters like screen_name or count.
      max_tries .. The maximum number of tries to attempt.
    Returns:
      A TwitterResponse object, or None if failed.
    """
    for i in range(max_tries):
        request = twitter.request(resource, params)
        #print(request.status_code)        
        if request.status_code == 200:
            return request
        else:
            print('Got error %s \nsleeping for 15 minutes.' % request.text)
            sys.stderr.flush()
            time.sleep(61 * 15)


def get_users(twitter, screen_names):
    """Retrieve the Twitter user objects for each screen_name.
    Params:
        twitter........The TwitterAPI object.
        screen_names...A list of strings, one per screen_name
    Returns:
        A list of dicts, one per user, containing all the user information
        (e.g., screen_name, id, location, etc)
    See the API documentation here: https://dev.twitter.com/rest/reference/get/users/lookup
    In this example, I test retrieving two users: twitterapi and twitter.
    >>> twitter = get_twitter()
    >>> users = get_users(twitter, ['twitterapi', 'twitter'])
    >>> [u['id'] for u in users]
    [6253282, 783214]
    """
    ###TODO       
    print(screen_names)
    users = robust_request(twitter,"users/lookup",{'screen_name':screen_names},5)        
    return users     



def get_friends(twitter, screen_name):
    """ Return a list of Twitter IDs for users that this person follows, up to 5000.
    See https://dev.twitter.com/rest/reference/get/friends/ids
    Note, because of rate limits, it's best to test this method for one candidate before trying
    on all candidates.
    Args:
        twitter.......The TwitterAPI object
        screen_name... a string of a Twitter screen name
    Returns:
        A list of ints, one per friend ID, sorted in ascending order.
    Note: If a user follows more than 5000 accounts, we will limit ourselves to
    the first 5000 accounts returned.
    In this test case, I return the first 5 accounts that I follow.
    >>> twitter = get_twitter()
    >>> get_friends(twitter, 'aronwc')[:5]
    [695023, 1697081, 8381682, 10204352, 11669522]
    """
    ###TODO
    
    
    friends = robust_request(twitter,"friends/ids",{'screen_name':screen_name,'count':5000,'cursor':-1},5)
    #print("Type of friends:",type(friends))
    frn = [r['ids'] for r in friends]    
        
    for a in frn:
        a=sorted(a)
        #print("List of friends are :",len(a))       
    
    return a

    

def add_all_friends(twitter, users):
    """ Get the list of accounts each user follows.
    I.e., call the get_friends method for all 4 candidates.
    Store the result in each user's dict using a new key called 'friends'.
    Args:
        twitter...The TwitterAPI object.
        users.....The list of user dicts.
    Returns:
        Nothing
    >>> twitter = get_twitter()
    >>> users = [{'screen_name': 'aronwc'}]
    >>> add_all_friends(twitter, users)
    >>> users[0]['friends'][:5]
    [695023, 1697081, 8381682, 10204352, 11669522]
    """
    ###TODO
    u = [x['screen_name'] for x in users]    
    for i in range(len(users)):
        name=users[i]['screen_name']
        friends_list = get_friends(twitter, name)
        users[i]['friends']=friends_list


def print_num_friends(users):
    """Print the number of friends per candidate, sorted by candidate name.
    See Log.txt for an example.
    Args:
        users....The list of user dicts.
    Returns:
        Nothing
    """
    ###TODO
    sorted_list=sorted(users, key=lambda x: x['screen_name'])

    for i in range(len(sorted_list)):
        print(sorted_list[i]['screen_name'] , len(sorted_list[i]['friends']))


def count_friends(users):
    """ Count how often each friend is followed.
    Args:
        users: a list of user dicts
    Returns:
        a Counter object mapping each friend to the number of candidates who follow them.
        Counter documentation: https://docs.python.org/dev/library/collections.html#collections.Counter
    In this example, friend '2' is followed by three different users.
    >>> c = count_friends([{'friends': [1,2]}, {'friends': [2,3]}, {'friends': [2,3]}])
    >>> c.most_common()
    [(2, 3), (3, 2), (1, 1)]
    """
    ###TODO        
    c= Counter()
    for i in users:
        c.update(i['friends'])
    return c    


def friend_overlap(users):
    """
    Compute the number of shared accounts followed by each pair of users.
    Args:
        users...The list of user dicts.
    Return: A list of tuples containing (user1, user2, N), where N is the
        number of accounts that both user1 and user2 follow.  This list should
        be sorted in descending order of N. Ties are broken first by user1's
        screen_name, then by user2's screen_name (sorted in ascending
        alphabetical order). See Python's builtin sorted method.
    In this example, users 'a' and 'c' follow the same 3 accounts:
    >>> friend_overlap([
    ...     {'screen_name': 'a', 'friends': ['1', '2', '3']},
    ...     {'screen_name': 'b', 'friends': ['2', '3', '4']},
    ...     {'screen_name': 'c', 'friends': ['1', '2', '3']},
    ...     ])
    [('a', 'c', 3), ('a', 'b', 2), ('b', 'c', 2)]
    """
    ###TODO    
    frnd_overlap=[]
    icount=0        
    for i in users:
        icount+=1
        jcount=0
        for j in users:
            jcount+=1
            if (icount<jcount):             
                fname = i['screen_name']
                sname = j['screen_name']                
                c=Counter()
                c.update(i['friends'])
                c.update(j['friends'])  
                common_friends = [k for k in i['friends'] if c[k]==2]                           
                frnd_overlap.append((fname,sname,len(common_friends)))

    frnd_overlap=sorted(frnd_overlap, key=lambda x: (-x[2],x[0],x[1]))
    return frnd_overlap


def followed_by_hillary_and_donald(users, twitter):
    """
    Find and return the screen_name of the one Twitter user followed by both Hillary
    Clinton and Donald Trump. You will need to use the TwitterAPI to convert
    the Twitter ID to a screen_name. See:
    https://dev.twitter.com/rest/reference/get/users/lookup
    Params:
        users.....The list of user dicts
        twitter...The Twitter API object
    Returns:
        A string containing the single Twitter screen_name of the user
        that is followed by both Hillary Clinton and Donald Trump.
    """
    ###TODO\    
    for i in users:
        if i['screen_name']=="HillaryClinton":          
            for j in users:             
                if j['screen_name']=="realDonaldTrump":                 
                    c=Counter()
                    c.update(i['friends'])
                    c.update(j['friends'])
                    common_friend=[k for k in i['friends'] if c[k]==2]
                    break
    #print(common_friend)    
    name = robust_request(twitter,"users/lookup",{'user_id':common_friend},5)
    li = [r for r in name]
    return li[0]['screen_name']


def create_graph(users, friend_counts):
    """ Create a networkx undirected Graph, adding each candidate and friend
        as a node.  Note: while all candidates should be added to the graph,
        only add friends to the graph if they are followed by more than one
        candidate. (This is to reduce clutter.)
        Each candidate in the Graph will be represented by their screen_name,
        while each friend will be represented by their user id.
    Args:
      users...........The list of user dicts.
      friend_counts...The Counter dict mapping each friend to the number of candidates that follow them.
    Returns:
      A networkx Graph
    """
    ###TODO
    graph = nx.Graph()
    for i in users:
        candidate_name= i['screen_name']
        for j in i['friends']:
            for key in friend_counts:
                if (j == key and friend_counts[key]>1):                 
                    graph.add_edge(i['screen_name'],j)
                    break
    return graph


def draw_network(graph, users, filename):
    """
    Draw the network to a file. Only label the candidate nodes; the friend
    nodes should have no labels (to reduce clutter).
    Methods you'll need include networkx.draw_networkx, plt.figure, and plt.savefig.
    Your figure does not have to look exactly the same as mine, but try to
    make it 9look presentable.
    """
    ###TODO
    pos=nx.spring_layout(graph)
    #plt.figure()
    labels={}
    for i in users:
        labels[i['screen_name']]=i['screen_name']
    #print(labels)  
    nx.draw_networkx(graph,pos,with_labels=False,node_color='blue',node_size=50,alpha=0.50,edge_color='r')
    nx.draw_networkx_labels(graph,pos,labels,font_size=12,figsize=(12,12))
    #plt.figure(graph,figsize=(12,12))
    plt.axis('off')
    plt.savefig(filename,format="PNG",frameon=None,dpi=500)
    plt.show()   



def main():


    """ Main method. You should not modify this. """
    twitter = get_twitter()    
    screen_names = read_screen_names('candidates.txt')    
    print('Established Twitter connection.')
    print('Read screen names: %s' % screen_names)
    users = sorted(get_users(twitter, screen_names), key=lambda x: x['screen_name']) 
    print('found %d users with screen_names %s' %
          (len(users), str([u['screen_name'] for u in users])))
    add_all_friends(twitter, users)
    print('Friends per candidate:')    
    print_num_friends(users)
    friend_counts = count_friends(users)    
    print('Most common friends:\n%s' % str(friend_counts.most_common(5)))    
    print('Friend Overlap:\n%s' % str(friend_overlap(users)))
    print('User followed by Hillary and Donald: %s' % followed_by_hillary_and_donald(users, twitter))
    graph = create_graph(users, friend_counts)
    print('graph has %s nodes and %s edges' % (len(graph.nodes()), len(graph.edges())))   
    draw_network(graph, users, 'network.png')
    print('network drawn to network.png')

if __name__ == '__main__':
    main()

# That's it for now! This should give you an introduction to some of the data we'll study in this course.