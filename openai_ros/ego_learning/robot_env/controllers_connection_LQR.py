#!/usr/bin/env python

import rospy
import time
from controller_manager_msgs.srv import SwitchController, SwitchControllerRequest, SwitchControllerResponse, LoadController, LoadControllerRequest, UnloadController, UnloadControllerRequest

from std_msgs.msg import Float64MultiArray


class ControllersConnection():

    def __init__(self, namespace, controllers_list):

        rospy.loginfo("Start Init ControllersConnection")
        self.controllers_list = controllers_list
        self.switch_service_name = "/ego/controller_manager/switch_controller"
        self.switch_service = rospy.ServiceProxy(
            "/ego/controller_manager/switch_controller", SwitchController)

        self._load_srv = rospy.ServiceProxy(
            '/ego/controller_manager/load_controller', LoadController)
        self._unload_srv = rospy.ServiceProxy(
            '/ego/controller_manager/unload_controller', UnloadController)
        rospy.loginfo("END Init ControllersConnection")
        self.publisherQBS_1 = rospy.Publisher(
            "/AlterEgo/reference_1", Float64MultiArray, queue_size=1)
        self.publisherQBS_2 = rospy.Publisher(
            "/AlterEgo/reference_2", Float64MultiArray, queue_size=1)
        self.initialQbs = Float64MultiArray()
        self.initialQbs.data = [0, 0, 0,
                                0, 0]
        self.initialQbs = Float64MultiArray()
        self.publisherQBS_1.publish(self.initialQbs)
        self.publisherQBS_2.publish(self.initialQbs)


    def _load_controller(self, name):
        self._load_srv.call(LoadControllerRequest(name=name))

    def switch_controllers(self, controllers_on, controllers_off, strictness=1):
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

    def reset_controllers(self):
        """
        We turn on and off the given controllers
        :param controllers_reset: ["name_controler_1", "name_controller2",...,"name_controller_n"]
        :return:
        """

        reset_result = False

        result_off_ok = self.switch_controllers(controllers_on=[],
                                                controllers_off=self.controllers_list)

        # rospy.logdebug("Deactivated Controllers")

        if result_off_ok:
            # rospy.logdebug("Activating Controllers")
            result_on_ok = self.switch_controllers(controllers_on=self.controllers_list,
                                                   controllers_off=[])
            if result_on_ok:
                # rospy.logdebug("Controllers Reseted==>"+str(self.controllers_list))
                reset_result = True
            else:
                rospy.logdebug("result_on_ok==>" + str(result_on_ok))
        else:
            rospy.logdebug("result_off_ok==>" + str(result_off_ok))

        return reset_result

    #AndCav
    def switch_off(self):
        result_off_ok = False
        pippo = rospy.wait_for_service(
            "/ego/controller_manager/switch_controller")
        result_off_ok = self.switch_controllers(controllers_on=[],
                                                controllers_off=['joint_state_controller', 'left_wheel_controller', 'right_wheel_controller'])
        if result_off_ok:
            # rospy.logdebug("Activating Controllers")

            #rospy.wait_for_service("/ego/controller_manager/unload_controller")
            unloadreq = UnloadControllerRequest()
            rospy.sleep(1)
            for o in self.controllers_list:
                unloadreq.name = o
                try:
                    #print("try to unload: "+unloadreq.name)
                    self._unload_srv.call(unloadreq)
                except rospy.ServiceException as e:
                    print(unloadreq+" service unloadcontroller call failed")
            # unloadreq.name = "joint_state_controller"
            # good = self._unload_srv.call(UnloadControllerRequest('joint_state_controller'))

    #AndCav
    def load_controller(self):
        loadreq = LoadControllerRequest()
        rospy.wait_for_service("/ego/controller_manager/load_controller")
        for o in self.controllers_list:
            loadreq.name = o
            try:
                self._load_srv.call(loadreq)
            except rospy.ServiceException as e:
                print(loadreq+" service loadcontroller call failed")

    #AndCav

    def switch_on(self):
        result_on_ok = False
        result_on_ok = self.switch_controllers(controllers_on=self.controllers_list,
                                               controllers_off=[])
        if(result_on_ok):
            #print("startedControllers")
            pass
        return result_on_ok

    def set_controllers_on(self):
        """
        We turn on and off the given controllers
        :param controllers_reset: ["name_controler_1", "name_controller2",...,"name_controller_n"]
        :return:
        """
        reset_result = False
        result_on_ok = self.switch_controllers(controllers_on=self.controllers_list,
                                               controllers_off=[])
        if result_on_ok:
            reset_result = True
        else:
            rospy.logdebug("result_on_ok==>" + str(result_on_ok))

        return reset_result


    def update_controllers_list(self, new_controllers_list):
        self.controllers_list = new_controllers_list
