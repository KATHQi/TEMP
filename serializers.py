# -*- coding: utf-8 -*-
"""
----------------------------------------------------------------------
    File Name:  serializers
    Description:    
    date:  2022/6/2 0002 10:31
----------------------------------------------------------------------
"""
from rest_framework import serializers

from .models import Project, Picture, Mark, Moudeltype, Markresults, Exportresult_bulk


class PictureSerializer(serializers.ModelSerializer):

    class Meta:
        model = Picture
        fields = '__all__'


class ProjectSerializer(serializers.ModelSerializer):
    # image = ImageSerializer(many=True)

    class Meta:
        model = Project
        fields = '__all__'

class MoudeltypeSerializer(serializers.ModelSerializer):
    # image = ImageSerializer(many=True)

    class Meta:
        model = Moudeltype
        fields = '__all__'

class MarkSerializer(serializers.ModelSerializer):

    class Meta:
        model = Mark
        fields = '__all__'
class MarkresultsSerializer(serializers.ModelSerializer):

    class Meta:
        model = Markresults
        fields = '__all__'

class Exportresult_bulkSerializer(serializers.ModelSerializer):

    class Meta:
        model = Exportresult_bulk
        fields = '__all__'


class ProjectWithImage(serializers.ModelSerializer):
    picture = PictureSerializer(many=False)

    class Meta:
        model = Project
        # fields = ['project_name','project_desc', 'image']
        fields = '__all__'
class ModelWithImage(serializers.ModelSerializer):
    picture = PictureSerializer(many=False)

    class Meta:
        model = Moudeltype
        # fields = ['project_name','project_desc', 'image']
        fields = '__all__'


class ProjectWithMark(serializers.ModelSerializer):
    mark_set = MarkSerializer(many=True)

    class Meta:
        model = Project
        # fields = ['project_name','project_desc', 'mark']
        fields = '__all__'

