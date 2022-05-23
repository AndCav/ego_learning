import rospy
import time
from std_srvs.srv import Empty
from gazebo_msgs.srv import SetPhysicsProperties, SetPhysicsPropertiesRequest, DeleteModel, DeleteModelRequest,  SpawnModel, SpawnModelRequest, SetModelConfiguration, SetModelConfigurationRequest, GetModelState,GetModelPropertiesResponse
from std_msgs.msg import Float64
from geometry_msgs.msg import Vector3
from ego_learning.msg import RLExperimentInfo
import subprocess
import roslaunch
class RespawnerC():

    def __init__(self,first):
        self.last_update=first
        self.counter=0
        self.subber=rospy.Subscriber("/openai/reward",RLExperimentInfo,self.callback)
        self.rate = rospy.Rate(1) # 10hz
        self.period=25
    def callback(self,data):
        self.last_update=rospy.get_time()
        
    

    def respawner_check(self):
        
        while not rospy.is_shutdown():
            if(rospy.get_time()-self.last_update)>=self.period:
                #pub.publish(hello_str)rospy.on_shutdown(self.shutdown)

                uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
                roslaunch.configure_logging(uuid)
                launch = roslaunch.parent.ROSLaunchParent(uuid,["/home/cp/RL_ws/src/ego_learning/launch/ego_gazebo_noPlugin_reset.launch"])
                launch.start()
                rospy.sleep(3)
                launch.shutdown()
                self.last_update=rospy.get_time()
                self.counter+=1
                print("too much time from last spawning. restarted {:d} times".format(self.counter))
            self.rate.sleep()

if __name__ == '__main__':

    rospy.init_node('listener', anonymous=True)
    res=RespawnerC(rospy.get_time())
    res.respawner_check()
    