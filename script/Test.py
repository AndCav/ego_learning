import rospy
import time
from controller_manager_msgs.srv import SwitchController, SwitchControllerRequest, LoadController, LoadControllerRequest, UnloadController, UnloadControllerRequest
from std_srvs.srv import Empty
from gazebo_msgs.msg import ODEPhysics
from gazebo_msgs.srv import SetPhysicsProperties, SetPhysicsPropertiesRequest, DeleteModel, DeleteModelRequest,  SpawnModel, SpawnModelRequest, SetModelConfiguration, SetModelConfigurationRequest, GetModelState,GetModelPropertiesResponse
from std_msgs.msg import Float64
from geometry_msgs.msg import Vector3
import roslaunch
from roslaunch import rlutil, parent
import os

class GazeboConnection():

    def __init__(self, start_init_physics_parameters, reset_world_or_sim, max_retry=100):
        self.paused_done=True
        self.unpaused_done=True
        self._max_retry = max_retry
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_simulation_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        self.reset_world_proxy = rospy.ServiceProxy('/gazebo/reset_world', Empty)


        #personal edits
        self.uuid = rlutil.get_or_generate_uuid(None, False)
        roslaunch.configure_logging(self.uuid)
        self.reset_launch = parent.ROSLaunchParent(
            self.uuid, ["/home/cp/RL_ws/src/ego_learning/launch/ego_gazebo_noArms_reset.launch"])
        self.delete_model = rospy.ServiceProxy(
            '/gazebo/delete_model', DeleteModel)
        self.spawn_model = rospy.ServiceProxy(
            '/gazebo/spawn_urdf_model', SpawnModel)
        self.robotDesc = rospy.get_param("/robot_description")
        self.spawn_req = SpawnModelRequest()
        self.spawn_req.model_xml = self.robotDesc
        self.spawn_req.initial_pose.position.x = 0.0
        self.spawn_req.initial_pose.position.y = 0.0
        self.spawn_req.initial_pose.position.z = 0.13
        self.spawn_req.model_name = str("ego_robot")
        self.robotDesc = rospy.get_param("/robot_description")
        self.set_model_proxy = rospy.ServiceProxy(
            '/gazebo/set_model_configuration', SetModelConfiguration)
        self.setmodel = SetModelConfigurationRequest()
        self.setmodel.model_name = str("ego_robot")
        self.setmodel.urdf_param_name = self.robotDesc
        self.setmodel.joint_names = ['Wheel_L', 'Wheel_R']
        #rosservice call /gazebo/set_model_configuration '{model_name: "ego_robot", urdf_param_name: "robot_description", joint_names:['L_joint_baseW', 'R_joint_baseW'], joint_positions:[0.0, 0.0]}'
        self.setmodel.joint_positions = [0.0, 0.0]
        # Setup the Gravity Controle system
        service_name = '/gazebo/set_physics_properties'
        rospy.logdebug("Waiting for service " + str(service_name))
        rospy.wait_for_service(service_name)
        rospy.logdebug("Service Found " + str(service_name))
        self.set_physics = rospy.ServiceProxy(
            service_name, SetPhysicsProperties)
        self.start_init_physics_parameters = start_init_physics_parameters
        self.reset_world_or_sim = "WORLD"
        self.init_values()
        # We always pause the simulation, important for legged robots learning
        #self.pauseSim()

    def pauseSim(self):
        # rospy.logdebug("PAUSING service found...")
        self.paused_done = False
        counter = 0
        while not self.paused_done and not rospy.is_shutdown():
            if counter < self._max_retry:
                try:
                    # rospy.logdebug("PAUSING service calling...")
                    self.pause()
                    self.unpaused_done = False
                    self.paused_done = True
                    # rospy.logdebug("PAUSING service calling...DONE")
                except rospy.ServiceException as e:
                    counter += 1
                    rospy.logerr("/gazebo/pause_physics service call failed")
            else:
                error_message = "Maximum retries done" + \
                    str(self._max_retry)+", please check Gazebo pause service"
                rospy.logerr(error_message)
                assert False, error_message

        # rospy.logdebug("PAUSING FINISH")

    def unpauseSim(self):
        # rospy.logdebug("UNPAUSING service found...")
        self.unpaused_done = False
        counter = 0
        while not self.unpaused_done and not rospy.is_shutdown():
            if counter < self._max_retry:
                try:
                    # rospy.logdebug("UNPAUSING service calling...")
                    self.unpause()
                    self.paused_done = False
                    self.unpaused_done = True
                    # rospy.logdebug("UNPAUSING service calling...DONE")
                except rospy.ServiceException as e:
                    counter += 1
                    rospy.logerr(
                        "/gazebo/unpause_physics service call failed...Retrying "+str(counter))
            else:
                error_message = "Maximum retries done" + \
                    str(self._max_retry)+", please check Gazebo unpause service"
                rospy.logerr(error_message)
                assert False, error_message

        # rospy.logdebug("UNPAUSING FiNISH")

    def setconfiguration(self):
        self.set_model_proxy(self.setmodel)

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
            #self.safeReset()

        elif self.reset_world_or_sim == "NO_RESET_SIM":
            rospy.logerr("NO RESET SIMULATION SELECTED")
        else:
            rospy.logerr("WRONG Reset Option:"+str(self.reset_world_or_sim))

    def resetSimulation(self):
        rospy.wait_for_service('/gazebo/reset_simulation')
        try:
            self.reset_simulation_proxy()
        except rospy.ServiceException as e:
            print("/gazebo/reset_simulation service call failed")

    def resetWorld(self):
        rospy.wait_for_service('/gazebo/reset_world')
        print(self.reset_world_or_sim)
        try:
            self.reset_world_proxy()

        except rospy.ServiceException as e:
            print("/gazebo/reset_world service call failed")

    def deleteModel_(self):  # AndCav
        """
        the only way to avoid the undesired stop of the /ego/joint_state topic is the following:
        -pause the simulation
        -deletemodel
        -reset the simulation (along with simulationTime)
        -relaunch all the controller and the urdf spawner (forcing a reset)
        -unpause simulation
        """

        pippo=rospy.wait_for_service("/ego/controller_manager/switch_controller")
        print("trying to delete ego_robot")
        test = "ego_robot"
        delete_req = DeleteModelRequest()
        delete_req.model_name = test
        self.delete_model(delete_req)
        #os.system("roslaunch ego_learning ego_gazebo_noArms_reset.launch")
        # self.spawn_model(self.spawn_req)
        # self.setconfiguration()
        #self.reset_launch.start()
        #self.unpauseSim()

    def respawn(self):
        pippo=self.spawn_model.call(self.spawn_req)
        # self.setconfiguration()

    def init_values(self):

        self.resetSim()

        if self.start_init_physics_parameters:
            rospy.logdebug("Initialising Simulation Physics Parameters")
            self.init_physics_parameters()
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
        rospy.logdebug("Gravity Update Result==" + str(result.success) +
                       ",message==" + str(result.status_message))

        self.unpauseSim()

    def change_gravity(self, x, y, z):
        self._gravity.x = x
        self._gravity.y = y
        self._gravity.z = z

        self.update_gravity_call()


class ControllersConnection():

    def __init__(self, namespace, controllers_list):

        rospy.loginfo("Start Init ControllersConnection")
        self.controllers_list = controllers_list
        self.switch_service_name = '/ego/controller_manager/switch_controller'
        self.switch_service = rospy.ServiceProxy(
            self.switch_service_name, SwitchController)
        self.load_service_name = '/ego/controller_manager/load_controller'
        self._load_srv = rospy.ServiceProxy(
            self.load_service_name, LoadController)
        self.unload_service_name = '/ego/controller_manager/unload_controller'
        self._unload_srv = rospy.ServiceProxy(
            self.unload_service_name, UnloadController)
        #print(self.switch_service_name)
        rospy.loginfo("END Init ControllersConnection")

    def _load_controller(self, name):
        self._load_srv.call(LoadControllerRequest(name=name))

    def switch_controllers(self, controllers_on, controllers_off, strictness=2):
        """
        Give the controllers you want to switch on or off.
        :param controllers_on: ["name_controler_1", "name_controller2",...,"name_controller_n"]
        :param controllers_off: ["name_controler_1", "name_controller2",...,"name_controller_n"]
        :return:
        """

        rospy.wait_for_service(self.switch_service_name)

        try:
            switch_request_object = SwitchControllerRequest()
            switch_request_object.start_controllers = controllers_on
            switch_request_object.stop_controllers = controllers_off
            switch_request_object.strictness = strictness

            switch_result = self.switch_service(switch_request_object)
            """
            [controller_manager_msgs/SwitchController]
            int32 BEST_EFFORT=1
            int32 STRICT=2
            string[] start_controllers
            string[] stop_controllers
            int32 strictness
            ---
            bool ok
            """
            rospy.logdebug("Switch Result==>"+str(switch_result.ok))
            return switch_result.ok

        except rospy.ServiceException as e:
            print(self.switch_service_name+" service call failed")

            return None

    #AndCav
    def switch_off(self):
        result_off_ok = False
        pippo=rospy.wait_for_service("/ego/controller_manager/switch_controller")
        result_off_ok = self.switch_controllers(controllers_on=[],
                                                controllers_off=['joint_state_controller','left_wheel_controller','right_wheel_controller'])
        if result_off_ok:
            # rospy.logdebug("Activating Controllers")
            unloadreq = UnloadControllerRequest()
            for o in self.controllers_list:
                unloadreq.name = o
                try:
                    print("try to unload: "+unloadreq.name)
                    good = self._unload_srv.call(unloadreq)
                    if(good):
                        print("unloaded"+o)
                        good=False
                except rospy.ServiceException as e:
                    print(unloadreq+" service unloadcontroller call failed")
            # unloadreq.name = "joint_state_controller"
            # good = self._unload_srv.call(UnloadControllerRequest('joint_state_controller'))

    #AndCav
    def load_controller(self):
        loadreq = LoadControllerRequest()
        for o in self.controllers_list:
            loadreq.name = o
            good = self._load_srv.call(loadreq)
            if(good):
                print("loaded"+o)
                good=False

    def switch_on(self):
        
        # loadreq.name = "joint_state_controller"
        # good = self._load_srv.call(LoadControllerRequest('joint_state_controller'))
        result_on_ok =False
        result_on_ok = self.switch_controllers(controllers_on=self.controllers_list,
                                               controllers_off=[])
        if(result_on_ok):
            print("startedControllers")
        return result_on_ok

    def reset_controllers(self):
        """
        We turn on and off the given controllers
        :param controllers_reset: ["name_controler_1", "name_controller2",...,"name_controller_n"]
        :return:
        """
        reset_result = False

        # result_off_ok = self.switch_controllers(controllers_on=[],
        #                                         controllers_off=self.controllers_list)
        result_off_ok = self.switch_off()
        rospy.logdebug("Deactivated Controllers")

        if result_off_ok:
            # rospy.logdebug("Activating Controlers")
            # unloadreq = UnloadControllerRequest()
            # for o in self.controllers_list:
            #     unloadreq.name = o
            #     self._unload_srv.call(unloadreq)

            # result_on_ok = self.switch_controllers(controllers_on=self.controllers_list,
            #                                        controllers_off=[])
            result_on_ok=self.switch_on()
            if result_on_ok:
                rospy.logdebug("Controllers Reseted==>"+str(self.controllers_list))
                reset_result = True
            else:
                rospy.logdebug("result_on_ok==>" + str(result_on_ok))
        else:
            rospy.logdebug("result_off_ok==>" + str(result_off_ok))

        return reset_result

    # def set_controllers_on(self):
    #     """
    #     We turn on and off the given controllers
    #     :param controllers_reset: ["name_controler_1", "name_controller2",...,"name_controller_n"]
    #     :return:
    #     """
    #     reset_result = False
    #     result_on_ok = self.switch_controllers(controllers_on=self.controllers_list,
    #                                            controllers_off=[])
    #     if result_on_ok:
    #         # rospy.logdebug("Controllers Reseted==>"+str(self.controllers_list))
    #         reset_result = True
    #     else:
    #         rospy.logdebug("result_on_ok==>" + str(result_on_ok))
    #         # rospy.logdebug("result_off_ok==>" + str(result_off_ok))

    #     return reset_result

    def update_controllers_list(self, new_controllers_list):
        self.controllers_list = new_controllers_list

if __name__ == "__main__":
    controllerTester = ControllersConnection(
        "ego", ['joint_state_controller', 'left_wheel_controller', 'right_wheel_controller'])
    gazeboTester = GazeboConnection(start_init_physics_parameters=True, reset_world_or_sim="SIMULATION")
    #tester.switch_controllers(controllers_on=[],controllers_off=["joint_state_controller", "left_wheel_controller", "right_wheel_controller"])
   
    # controllerTester.switch_off()
    gazeboTester.deleteModel_()
    gazeboTester.reset_simulation_proxy()
    print("switch_off")
    gazeboTester.respawn()
    gazeboTester.pauseSim()
    rospy.sleep(0.5)
    controllerTester.load_controller()
    rospy.sleep(0.5)
    gazeboTester.unpause()
    controllerTester.switch_on()



# controller.switch_off()
# gazebo.deleteModel_()
# gazebo.pauseSim()
# gazebo.reset_simulation_proxy()
# gazebo.respawn()
# rospy.sleep(2)
# controller.load_controller()
# rospy.sleep(0.5)
# gazebo.unpauseSim()
# controller.switch_on()

