#!/usr/bin/env python

from pyexpat import model
import rospy
from std_srvs.srv import Empty
from gazebo_msgs.msg import ODEPhysics
from gazebo_msgs.srv import SetPhysicsProperties, SetPhysicsPropertiesRequest, DeleteModel, DeleteModelRequest, SpawnModel, SpawnModelRequest, GetModelState, GetModelStateResponse, GetModelStateRequest
from std_msgs.msg import Float64
from geometry_msgs.msg import Vector3, Pose


class GazeboConnection():

    def __init__(self, start_init_physics_parameters, reset_world_or_sim, max_retry = 20):

        self._max_retry = max_retry
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_simulation_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        self.reset_world_proxy = rospy.ServiceProxy('/gazebo/reset_world', Empty)
 
        #personal edits
        self.delete_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
        self.spawn_model = rospy.ServiceProxy('/gazebo/spawn_urdf_model', SpawnModel)
        self.robotDesc = rospy.get_param("/robot_description")



        self.spawn_req = SpawnModelRequest()
        self.spawn_req.model_xml = self.robotDesc
        self.spawn_req.initial_pose.position.x = 0.0
        self.spawn_req.initial_pose.position.y = 0.0
        self.spawn_req.initial_pose.position.z = 0.13
        self.spawn_req.model_name = str("ego_robot")

        self.spawn_model_client = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        self.spawnsdf=SpawnModelRequest()
        self.spawnsdf.model_name=str("ego_robot")
        self.spawnsdf.model_xml=open('/home/cp/RL_ws/src/ego_learning/ego_robot/model.sdf', 'r').read()
        self.robot_namespace='/ego'
        self.spawnsdf.initial_pose.position.x=0
        self.spawnsdf.initial_pose.position.y=0
        self.spawnsdf.initial_pose.position.z=0.13


        test = "ego_robot"
        self.delete_req = DeleteModelRequest()
        self.delete_req.model_name = test
        
        # Setup the Gravity Controle system
        service_name = '/gazebo/set_physics_properties'
        rospy.logdebug("Waiting for service " + str(service_name))
        rospy.wait_for_service(service_name)
        rospy.logdebug("Service Found " + str(service_name))
        self.set_physics = rospy.ServiceProxy(service_name, SetPhysicsProperties)
        self.start_init_physics_parameters = start_init_physics_parameters
        self.reset_world_or_sim = reset_world_or_sim
        self.init_values()

        # We always pause the simulation, important for legged robots learning
        self.pauseSim()

    def pauseSim(self):
        # rospy.logdebug("PAUSING service found...")
        paused_done = False
        counter = 0
        while not paused_done and not rospy.is_shutdown():
            if counter < self._max_retry:
                try:
                    # rospy.logdebug("PAUSING service calling...")
                    self.pause()
                    paused_done = True
                    # rospy.logdebug("PAUSING service calling...DONE")
                except rospy.ServiceException as e:
                    counter += 1
                    rospy.logerr("/gazebo/pause_physics service call failed")
            else:
                error_message = "Maximum retries done"+str(self._max_retry)+", please check Gazebo pause service"
                rospy.logerr(error_message)
                assert False, error_message

        # rospy.logdebug("PAUSING FINISH")

    def unpauseSim(self):
        # rospy.logdebug("UNPAUSING service found...")
        unpaused_done = False
        counter = 0
        while not unpaused_done and not rospy.is_shutdown():
            if counter < self._max_retry:
                try:
                    # rospy.logdebug("UNPAUSING service calling...")
                    self.unpause()
                    unpaused_done = True
                    # rospy.logdebug("UNPAUSING service calling...DONE")
                except rospy.ServiceException as e:
                    counter += 1
                    rospy.logerr("/gazebo/unpause_physics service call failed...Retrying "+str(counter))
            else:
                error_message = "Maximum retries done"+str(self._max_retry)+", please check Gazebo unpause service"
                rospy.logerr(error_message)
                assert False, error_message

        # rospy.logdebug("UNPAUSING FiNISH")
  
    def resetSim(self):
        """
        This was implemented because some simulations, when reseted the simulation
        the systems that work with TF break, and because sometime we wont be able to change them
        we need to reset world that ONLY resets the object position, not the entire simulation
        systems.
        """
        if self.reset_world_or_sim == "SIMULATION":
            # rospy.logerr("SIMULATION RESET")
            self.resetSimulation()
        elif self.reset_world_or_sim == "WORLD":
            # rospy.logerr("WORLD RESET")
            self.resetWorld()
        elif self.reset_world_or_sim == "NO_RESET_SIM":
            rospy.logerr("NO RESET SIMULATION SELECTED")
        else:
            rospy.logerr("WRONG Reset Option:"+str(self.reset_world_or_sim))

    def resetSimulation(self):
        rospy.wait_for_service('/gazebo/reset_simulation')
        try:
            self.reset_simulation_proxy()
        except rospy.ServiceException as e:
            print ("/gazebo/reset_simulation service call failed")

    def resetWorld(self):
        rospy.wait_for_service('/gazebo/reset_world')
        try:
            self.reset_world_proxy()
        except rospy.ServiceException as e:
            print ("/gazebo/reset_world service call failed")
   #AndCav
    def deleteModel_(self):


        pippo=rospy.wait_for_service(str("/gazebo/delete_model"))
        #print("deleting_model")
        self.delete_model(self.delete_req)
        #print("deleted_model")

  # AndCav
    def respawn(self):
        rospy.wait_for_service("gazebo/spawn_urdf_model")
        self.spawn_model.call(self.spawn_req)
        gms = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        model_msg=None

        model_msg=gms("ego_robot","")
        if(model_msg!=None):
            model_msg.success ==False
        count=0
        while model_msg.success == False or model_msg==None:
            model_msg = gms("ego_robot","")
            count+=1
            if count ==5:
                self.spawn_model.call(self.spawn_req,)
                count = 0
        

        # modelstate_try=rospy.ServiceProxy(str("gazebo/get_model_state"))
        # modelstate_try.call
        # while not response:
        #     pass

    def init_values(self):

        self.resetSim()

        if self.start_init_physics_parameters:
            rospy.logdebug("Initialising Simulation Physics Parameters")
            #self.init_physics_parameters()
        else:
            rospy.logerr("NOT Initialising Simulation Physics Parameters")

    def init_physics_parameters(self):
        """
        We initialise the physics parameters of the simulation, like gravity,
        friction coeficients and so on.
        """
        self._time_step = Float64(0.001)
        self._max_update_rate = Float64(0.0)

        self._gravity = Vector3()
        self._gravity.x = 0.0
        self._gravity.y = 0.0
        self._gravity.z = -9.81

        self._ode_config = ODEPhysics()
        self._ode_config.auto_disable_bodies = False
        self._ode_config.sor_pgs_precon_iters = 0
        self._ode_config.sor_pgs_iters = 50
        self._ode_config.sor_pgs_w = 1.3
        self._ode_config.sor_pgs_rms_error_tol = 0.0
        self._ode_config.contact_surface_layer = 0.001
        self._ode_config.contact_max_correcting_vel = 0.0
        self._ode_config.cfm = 0.0
        self._ode_config.erp = 0.2
        self._ode_config.max_contacts = 20

        self.update_gravity_call()


    def update_gravity_call(self):

        self.pauseSim()

        set_physics_request = SetPhysicsPropertiesRequest()
        set_physics_request.time_step = self._time_step.data
        set_physics_request.max_update_rate = self._max_update_rate.data
        set_physics_request.gravity = self._gravity
        set_physics_request.ode_config = self._ode_config

        rospy.logdebug(str(set_physics_request.gravity))

        result = self.set_physics(set_physics_request)
        rospy.logdebug("Gravity Update Result==" + str(result.success) + ",message==" + str(result.status_message))

        self.unpauseSim()

    def change_gravity(self, x, y, z):
        self._gravity.x = x
        self._gravity.y = y
        self._gravity.z = z

        self.update_gravity_call()